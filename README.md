# dog_feel_classify_light  

   動画 で簡単分類、おそるべし Video Transformer.  
  
[画像と音（スペクトログラム）で分類、マルチモーダルTransformer.](https://www.netosa.com/blog/2026/04/transformer.html)  
の続きです。  
  
CNN-Transformer ハイブリッドモデル で、動画のクラス分類をする。  
かって、CNNとLSTMを組み合わせたモデルの「LRCN (Long-term Recurrent Convolutional Networks)」の、 LSTM 部分を、  
Transformer に置き換えたモデル。  
  
犬の動画(今回は、映像部分のみ) を使って、わんこの気持ちを予測します。  

##### 1. Train  
  dog_feel_light_train.ipynb  

##### 2. Onnx 変換  
  
  $ python dog_feel_light_torch2onnx_for_pc.py  

##### 3. Onnx predict  

  $ python dog_feel_light_orangepi_onnx.py  
