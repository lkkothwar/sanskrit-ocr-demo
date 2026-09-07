import os
import pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------
# 1. MODEL DEFINITIONS (unchanged)
# -------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x + self.gamma * (attention * x)

class MultiScaleInception(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.1):
        super().__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 32, 1, bias=False)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate))
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, padding=2, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate))
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, 32, 1, bias=False), nn.BatchNorm2d(32),
            ChannelAttention(32, ratio=8))
        self.output_proj = nn.Sequential(
            nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.shortcut = nn.Sequential()
        if in_channels != 128:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, 128, 1, bias=False), nn.BatchNorm2d(128))
        self.final_relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = self.shortcut(x)
        branches = [self.branch1x1(x), self.branch3x3(x), self.branch5x5(x), self.branch_pool(x)]
        concat = torch.cat(branches, dim=1)
        proj = self.output_proj(concat)
        out = proj + identity
        return self.final_relu(out)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_inception=True, dropout_rate=0.1):
        super().__init__()
        self.use_inception = use_inception
        if use_inception:
            self.conv1 = MultiScaleInception(in_channels, dropout_rate)
            mid_channels = 128
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
            mid_channels = out_channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or (use_inception and out_channels != mid_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.final_relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return self.final_relu(out)

class ShirorekhaNetFull(nn.Module):
    def __init__(self, in_channels=4, use_inception=True):
        super().__init__()
        self.enc1 = ResidualBlock(in_channels, 64, use_inception)
        self.enc2 = ResidualBlock(64, 128, use_inception)
        self.enc3 = ResidualBlock(128, 256, use_inception)
        self.enc4 = ResidualBlock(256, 512, use_inception)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(512, 512, use_inception)
        self.upconv4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(1024, 512, use_inception)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(512, 256, use_inception)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(256, 128, use_inception)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(128, 64, use_inception)
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1))
    def forward(self, x):
        e1 = self.enc1(x);   p1 = self.pool(e1)
        e2 = self.enc2(p1);  p2 = self.pool(e2)
        e3 = self.enc3(p2);  p3 = self.pool(e3)
        e4 = self.enc4(p3);  p4 = self.pool(e4)
        bn = self.bottleneck(p4)
        d4 = self.upconv4(bn); d4 = torch.cat([d4, e4], dim=1); d4 = self.dec4(d4)
        d3 = self.upconv3(d4); d3 = torch.cat([d3, e3], dim=1); d3 = self.dec3(d3)
        d2 = self.upconv2(d3); d2 = torch.cat([d2, e2], dim=1); d2 = self.dec2(d2)
        d1 = self.upconv1(d2); d1 = torch.cat([d1, e1], dim=1); d1 = self.dec1(d1)
        return self.out_conv(d1)

class AksharaNetEncoder(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.enc1 = ResidualBlock(in_channels, 64, use_inception=True, dropout_rate=0.1)
        self.enc2 = ResidualBlock(64, 128, use_inception=True, dropout_rate=0.1)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        return e2

class AksharaNet(nn.Module):
    def __init__(self, vocab_size, hidden=256, dropout=0.4):
        super().__init__()
        self.encoder = AksharaNetEncoder(in_channels=4)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, None))
        self.channel_compress = nn.Conv2d(128, 64, 1)
        self.rnn = nn.LSTM(128, hidden, bidirectional=True, num_layers=2, dropout=dropout, batch_first=False)
        self.fc = nn.Linear(hidden*2, vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x_raw, x_mask):
        x = torch.cat([x_raw, x_mask], dim=1)
        feat = self.encoder(x)
        feat = self.adaptive_pool(feat)
        feat = self.channel_compress(feat)
        B, C, H, W = feat.shape
        feat = feat.view(B, C*H, W).permute(2,0,1)
        feat = self.dropout(feat)
        out, _ = self.rnn(feat)
        logits = self.fc(out)
        return logits.permute(1,0,2)

# -------------------------------
# 2. INFERENCE ENGINE (SINGLETON)
# -------------------------------
class SanskritOCREngine:
    _instance = None

    def __new__(cls, model_dir="./models"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(model_dir)
        return cls._instance

    def _initialize(self, model_dir):
        # CRITICAL: Force CPU for HF Spaces free tier. 
        # If you later use a GPU Space, change this to 'cuda'.
        self.device = torch.device('cpu')
        
        # Prevent CPU thread oversubscription on shared HF runners
        torch.set_num_threads(1)
        print(f"[OCR] Loading models on {self.device}...")

        # Load char mapping
        mapping_path = os.path.join(model_dir, "char_mapping.pkl")
        with open(mapping_path, 'rb') as f:
            mapping = pickle.load(f)
            self.idx_to_char = mapping['idx_to_char']
            vocab_size = mapping['vocab_size']

        # Init models
        self.line_model = ShirorekhaNetFull(in_channels=4).to(self.device)
        self.ocr_model = AksharaNet(vocab_size=vocab_size).to(self.device)
        
        # Optional: Word model (skip if you don't need visualisations)
        # self.word_model = ShirorekhaNetFull(in_channels=4).to(self.device)

        # Load state dicts
        self._load_weights(self.line_model, os.path.join(model_dir, "ShirorekhaNet_line.pth"))
        self._load_weights(self.ocr_model, os.path.join(model_dir, "AksharaNet_best.pth"))
        
        # Set to eval mode
        self.line_model.eval()
        self.ocr_model.eval()
        print("[OCR] All models loaded successfully.")

    def _load_weights(self, model, path):
        # Force map_location='cpu' to avoid CUDA errors on HF Spaces
        ckpt = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        elif isinstance(ckpt, dict) and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in ckpt.items()):
            state = ckpt
        else:
            state = ckpt
        # strict=False is safer if there are minor mismatches
        model.load_state_dict(state, strict=False)

    # -------------------------------
    # 3. PREPROCESSING HELPERS (Paste yours here)
    # -------------------------------
    def _binarize(self, gray):
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    def _preprocess_full_page(self, img_rgb, target_size=512):
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        _, binary_orig = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_512 = cv2.resize(binary_orig, (target_size, target_size), interpolation=cv2.INTER_AREA)
        rgb_input = cv2.cvtColor(binary_512, cv2.COLOR_GRAY2RGB)
        bin_4th = self._binarize(binary_512)
        img_t = torch.from_numpy(rgb_input).permute(2,0,1).float() / 255.0
        bin_t = torch.from_numpy(bin_4th).unsqueeze(0).float() / 255.0
        input_t = torch.cat([img_t, bin_t], dim=0).unsqueeze(0).to(self.device)
        return input_t, binary_512

    def _predict_line_mask(self, input_tensor, threshold=0.5):
        with torch.no_grad():
            logits = self.line_model(input_tensor)
            if isinstance(logits, tuple): logits = logits[0]
            probs = torch.sigmoid(logits)
            mask = (probs > threshold).float().cpu().numpy()[0,0]
        return (mask * 255).astype(np.uint8)

    def _get_line_contours(self, mask_512, orig_shape, min_area=100):
        h_orig, w_orig = orig_shape
        mask_orig = cv2.resize(mask_512, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        valid.sort(key=lambda cnt: cv2.boundingRect(cnt)[1])
        return valid, mask_orig

    def _preprocess_line_for_ocr(self, line_bgr, line_mask_crop):
        rgb = cv2.cvtColor(line_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(rgb, (512, 64)).astype(np.float32) / 255.0
        mask = cv2.resize(line_mask_crop.astype(np.uint8), (512, 64),
                          interpolation=cv2.INTER_NEAREST).astype(np.float32)
        mask = (mask > 0.5).astype(np.float32)
        img_t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)
        return img_t, mask_t

    def _greedy_decode(self, logits_np):
        preds = logits_np.argmax(axis=-1)
        decoded, prev = [], 0  # blank idx = 0
        for idx in preds:
            idx = int(idx)
            if idx != 0 and idx != prev:
                decoded.append(self.idx_to_char.get(idx, '?'))
            prev = idx
        return ''.join(decoded)

    # -------------------------------
    # 4. MAIN INFERENCE LOOP
    # -------------------------------
    def process(self, image_bgr):
        h_orig, w_orig = image_bgr.shape[:2]
        rgb_orig = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Stage 1: Line Segmentation
        line_input, _ = self._preprocess_full_page(rgb_orig, target_size=512)
        line_mask_512 = self._predict_line_mask(line_input, threshold=0.5)
        line_contours, _ = self._get_line_contours(line_mask_512, (h_orig, w_orig))

        # Stage 2: OCR per line
        results = []
        for cnt in line_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            y1, y2 = max(0, y-3), min(h_orig, y+h+3)
            x1, x2 = x, x + w

            crop_original = image_bgr[y1:y2, x1:x2]  # BGR
            line_mask_roi = np.zeros((h_orig, w_orig), dtype=np.uint8)
            cv2.drawContours(line_mask_roi, [cnt], -1, 255, -1)
            line_mask_crop = line_mask_roi[y1:y2, x1:x2]

            img_t, mask_t = self._preprocess_line_for_ocr(crop_original, line_mask_crop)
            with torch.no_grad():
                logits = self.ocr_model(img_t, mask_t)
                logits_np = logits[0].cpu().numpy()
            text = self._greedy_decode(logits_np)
            results.append(text)

        return {
            "lines": results,
            "full_text": "\n".join(results)
        }