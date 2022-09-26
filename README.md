# 파이토치로 전이학습 분류 모델 만들기

## 패키지 설치
pip install -r requirements.txt

## 실행 예시
python train.py --model_fn resnet.pth --gpu_id 0 --n_epochs 20 --model_name resnet --n_classes 2 --freeze --use_pretrained
