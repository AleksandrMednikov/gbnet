# Модель решает задачу отнести скан к скану мозга или к скану легких
___
##### сылка на датасет 💿
* https://www.kaggle.com/datasets/darren2020/ct-to-mri-cgan 
* https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images
##### сылка на модель 💿
https://drive.google.com/drive/folders/1RDscpk8zdcyIcYcvT_BNmnXL4gjj_03b?usp=drive_link
##### образец данных на вход и выход 🔃
* вход: [матрица изображения RGB] | shape : (<кол-во изображений>, 128, 128, 3)
* выход: [[0.30, 0.70]] | shape : (<кол-во изображений>, 2)

##### метрики 📈
- prec : 
- rec : 
- f1 : 
- acc : 0.9889 | 0.993 порог: 0.6455790400505066  

##### матрица ошибок 🔢
| TP | FN |
|------|------|
| FP | TN |

##### Значение фаилов 📄
* preproc_makedata.ipynb : создание данных на вход(x_all.npy) и выход(y_all.npy) 
* search.ipynb : создание модели
* loadmodel.py : образец использования модели и указание препроцессинга

___
### Заметки ✏️
-
