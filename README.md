# 🏠 Klasifikasi Gambar Rumah Adat Indonesia

Proyek ini bertujuan untuk membangun model klasifikasi gambar rumah adat Indonesia menggunakan metode **Transfer Learning dengan MobileNetV2**, dan arsitektur tambahan berbasis **CNN (Conv2D dan Pooling)** dalam model Sequential.

## Datasets

from kaggle : https://www.kaggle.com/datasets/rariffirmansah/rumah-adat/data
from drive : https://drive.google.com/file/d/1bTGNfznwa4TDGrdPvSfqwcLnucDAN8a_/view?usp=sharing

## 🔍 Deskripsi Proyek

Indonesia memiliki keragaman budaya termasuk rumah adat dari berbagai daerah. Proyek ini mengklasifikasikan 5 jenis rumah adat menggunakan dataset gambar yang telah dibagi ke dalam folder:

- **Gadang** (Sumatera Barat)
- **Honai** (Papua)
- **Joglo** (Jawa Tengah)
- **Panjang** (Kalimantan Barat)
- **Tongkonan** (Sulawesi Selatan)

## 🧠 Model

Model dikembangkan dengan pendekatan:
- Arsitektur **Sequential** (Keras)
- Base model: `MobileNetV2` (tanpa top, pre-trained on ImageNet)
- Penambahan layer: `Conv2D`, `MaxPooling2D`, `Dropout`, dan `Dense`
- **Image size**: 224x224 px
- Optimizer: `Adamax`
- Loss Function: `categorical_crossentropy`

## 🧪 Preprocessing dan Augmentasi

```python
ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
