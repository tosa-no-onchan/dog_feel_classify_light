# dog_feel_light_orangepi_onnx.py
import numpy as np
import onnxruntime as ort
from PIL import Image
import time
import os

import cv2
import numpy as np

def resize_with_padding(image, target_size=(224, 224)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    # リサイズ
    resized = cv2.resize(image, (new_w, new_h))
    # 黒埋め用のキャンバス作成
    canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    # 中央に配置
    offset_y = (target_size[0] - new_h) // 2
    offset_x = (target_size[1] - new_w) // 2
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
    return canvas

def preprocess_images_numpy_old(frames):
    """ViTImageProcessorの完全再現 (NumPy版)"""
    # 0-255 -> 0-1 & Normalize (mean=0.5, std=0.5)
    # 計算式: (x / 255.0 - 0.5) / 0.5 => x / 127.5 - 1.0
    images = np.array(frames).astype(np.float32) / 127.5 - 1.0
    # (8, 224, 224, 3) -> (1, 8, 3, 224, 224) ※軸入れ替え含む
    images = images.transpose(0, 3, 1, 2)
    return np.expand_dims(images, axis=0)

def preprocess_images_numpy(frames):
    """ResNet18 (ImageNet) の前処理をNumPyで完全再現"""
    # 1. 0-255 (int) -> 0-1 (float32)
    images = np.array(frames).astype(np.float32) / 255.0
    
    # 2. ImageNetの正規化パラメータ
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    # 3. 正規化計算: (画像 - 平均) / 標準偏差
    # NumPyのブロードキャスト機能を利用
    images = (images - mean) / std
    
    # 4. 軸入れ替え: (Time, H, W, C) -> (Time, C, H, W)
    images = images.transpose(0, 3, 1, 2)
    
    # 5. バッチ次元追加: (1, Time, C, H, W)
    return np.expand_dims(images, axis=0)

class ONNXPredictor:
    def __init__(self, onnx_model_path, class_names, n_seconds=4, L_frames=8):
        # 1. ONNX Runtime セッションの作成
        # Orange Pi 5 の場合は 'CPUExecutionProvider' が基本ですが、
        # PCなら 'CUDAExecutionProvider' (GPU) も使えます。
        # ONNXセッションの初期化 (CPU専用)
        # Orange PiのCPUリソースをフル活用するため、セッションオプションを設定
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4  # Orange Piのコア数に合わせて調整
        self.session = ort.InferenceSession(
            onnx_model_path,
            sess_options=options, 
            providers=['CPUExecutionProvider'] 
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.class_names = class_names
        self.n_seconds = n_seconds
        self.L_frames = L_frames

        if False:
          self.padder = AspectRatioPad(size=(224, 224)) # 前回のカスタムクラス
          # 前処理 (Normalize)
          self.normalize = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          ])

    def predict(self, video_path):
        if False:
          # 2. 動画の読み込みとサンプリング
          video, _, _ = read_video(video_path, start_pts=0, end_pts=self.n_seconds, pts_unit='sec')
          total_frames = video.shape[0]

          indices = torch.linspace(0, total_frames - 1, steps=self.L_frames).long()
          sampled_video = video[indices]

          # 3. 前処理 (numpy配列として整形)
          processed_frames = []
          for frame in sampled_video:
              img = Image.fromarray(frame.numpy())
              img = self.padder(img)
              #img2 = resize_with_padding(img)
              img = self.normalize(img) # Tensor化 + 正規化
              processed_frames.append(img.numpy())

          # [1, Time, C, H, W] の形にする
          input_data = np.stack(processed_frames)[np.newaxis, ...]

        else:
          # --- 1. 映像読み込み (OpenCV) ---
          cap = cv2.VideoCapture(video_path)
          # (中略: indices計算と8フレーム抽出。以前のresize_with_paddingを使用)
          fps = cap.get(cv2.CAP_PROP_FPS) # 1秒あたりのフレーム数
          total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

          # --- 映像も「最初の3秒」に限定する ---
          #max_duration = 3.0
          # 3秒分、または動画全体の短い方のフレーム数をターゲットにする
          end_frame = min(total_frames, int(self.n_seconds * fps))

          # 0フレームから3秒地点（end_frame）の間で16枚抜く
          indices = np.linspace(0, end_frame - 1, self.L_frames).astype(int)

          frames = []
          for i in indices:
              cap.set(cv2.CAP_PROP_POS_FRAMES, i)
              ret, frame = cap.read()
              if ret:
                  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                  # 1. アスペクト比維持リサイズ
                  frame = resize_with_padding(frame)
                  frames.append(frame)
              else:
                  frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
          cap.release()
          # --- 2. 前処理 (NumPyのみ) ---
          input_data = preprocess_images_numpy(frames)
          #print('pixel_values.dtype:',pixel_values.dtype)

        # 4. ONNX 推論
        start_time = time.perf_counter()
        
        # 入力名をキーにした辞書でデータを渡す
        outputs = self.session.run(None, {self.input_name: input_data})
        
        inference_time = time.perf_counter() - start_time
        
        # 5. 結果の解析
        logits = outputs[0]
        pred_idx = np.argmax(logits, axis=1)[0]
        confidence = self._softmax(logits[0])[pred_idx]

        return self.class_names[pred_idx], confidence, inference_time

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


if __name__ == '__main__':
  MODEL_PATH = "/home/nishi/Documents/Visualstudio-torch_env/dog_feel_classify_light/dog_model_fixed-8_4.onnx"

  # --- 実行 ---
  #classes = ["cat", "dog", ...] # 学習時と同じ順序
  CLASS_NAMES =['background','alert', 'hungry', 'log_time_no_see', 'miss']


  predictor = ONNXPredictor(MODEL_PATH, CLASS_NAMES)


  # --- 1. ファイルパスとラベルのリストを作成 ---
  data_dir = "dataset_h264/miss"

  flist=os.listdir(data_dir)
  cnt=0

  import time

  # In[ ]:

  video_path=data_dir+'/'+flist[cnt]
  if True:
      for dir in CLASS_NAMES:
          data_dir = os.path.join("dataset_h264", dir)
          flist=os.listdir(data_dir)
          p_num = min(len(flist),100)
          print('-----')
          for i in range(p_num):
              video_path=data_dir+'/'+flist[i]
              print("video_path:",video_path)
              #cnt+=1
              #result, confidence=predict_video_fast(video_path, num_frames=num_frames, max_duration=max_duration)
              label, conf, t = predictor.predict(video_path)
              #print('result:',result, 'confidence:',confidence)
              print(f"結果: {label} ({conf:.2%})")
              print(f"ONNX推論時間: {t:.4f} 秒")



  #label, conf, t = predictor.predict("test_video.mp4")

  #print(f"結果: {label} ({conf:.2%})")
  #print(f"ONNX推論時間: {t:.4f} 秒")

