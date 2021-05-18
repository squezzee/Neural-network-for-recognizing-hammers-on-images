import imutils
import urllib.request
import numpy as np
import cv2
from PIL import Image
import random
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from keras.utils import to_categorical

# model zaczerpnięta z poniższego linka
# https://github.com/bnsreenu/python_for_microscopists/blob/master/208_multiclass_Unet_sandstone.py
# natknąłem się na nią oglądając ten tutorial: https://www.youtube.com/watch?v=XyX5HNuv-xE

# Co udało się zrobić:
# wygenerować 200 zdjęć na podstawie 10 losowych zdjęć z sieci
# stworzenie sieci i jej nauczenie na poziomie accuracy > 75%
# Dodać możliwość podawania zdjęcia z sieci na naszą sieć

# Nie działa wizualizacja nauki, miałem błąd związany z jakąś wtyczką którego nie zdążyem rozwiązać


def multi_unet_model(n_classes=2, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


# funkcja do przerabiania zdjęcia na zdjęcie z przezroczystym tłem
def transparent_image(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res, alpha = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY)
    blue, green, red = cv2.split(image)
    rgba = [blue, green, red, alpha]
    image_rgba = cv2.merge(rgba, 4)
    return image_rgba


# linki do zdjęć, przed Pani odpowiedzią na pytanie na discordzie tak to rozwiązałem, tj przy uruhodzmieniu programu pobieram zdjęcia z linków
links = ["https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg?crop=1.00xw:0.669xh;0,0.190xh&resize=980:*",
         "https://hips.hearstapps.com/ghk.h-cdn.co/assets/16/08/gettyimages-464163411.jpg?crop=1.0xw:1xh;center,top&resize=980:*",
         "https://hips.hearstapps.com/ghk.h-cdn.co/assets/17/30/pembroke-welsh-corgi.jpg?crop=1xw:0.9997114829774957xh;center,top&resize=980:*",
         "https://hips.hearstapps.com/ghk.h-cdn.co/assets/17/30/2560x3839/australian-shepherd.jpg?resize=980:*",
         "https://hips.hearstapps.com/ghk.h-cdn.co/assets/17/30/pit-bull.jpg?crop=1.0xw:1xh;center,top&resize=980:*",
         "https://hips.hearstapps.com/ghk.h-cdn.co/assets/16/08/gettyimages-462376265.jpg?crop=1.0xw:1xh;center,top&resize=980:*",
         "https://hips.hearstapps.com/ghk.h-cdn.co/assets/17/30/shetland-sheep-dog.jpg?crop=1.0xw:1xh;center,top&resize=980:*",
         "https://hips.hearstapps.com/ghk.h-cdn.co/assets/16/08/gettyimages-147786673.jpg?crop=0.4444444444444445xw:1xh;center,top&resize=980:*",
         "https://hips.hearstapps.com/ghk.h-cdn.co/assets/16/08/1280x1919/gettyimages-179494696.jpg?resize=980:*",
         "https://hips.hearstapps.com/ghk.h-cdn.co/assets/16/08/gettyimages-149263578.jpg?crop=0.4444444444444445xw:1xh;center,top&resize=980:*"]

images = []
names = []
# nazywam swoje zdjęcia pobrane z sieci
for i in range(0, 10):
    names.append("dog" + str(i) + ".png")
    urllib.request.urlretrieve(links[i], names[i])
    im = cv2.imread(names[i])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)
    images.append(im)

hammers = ["hammer1.png", "hammer2.png", "hammer3.png", "hammer4.png", "hammer5.png"]
transparent_hammers = []
# przetwarzam zdjęcia młotków tak aby ich tło było przezroczyste
for hammer in hammers:
    img_data = cv2.imread(hammer)
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    img_trans = transparent_image(img_data)
    transparent_hammers.append(img_trans)

# przygotowanie list odpowiednich zbiorów/mask
train_mask = []
train_dataset = []
val_mask = []
val_dataset = []
test_mask = []
test_dataset = []

# dla każdego zdjęcia losuje 20 razy młotek i jego pozycję, po czym obracam młotek i wklejam go na zdjęcie,
# po czym tworzę maskę, która odpowiada naszemu obrazowi z wklejonym młotkiem
for image in images:
    image_dog = Image.fromarray(image)
    temp_mask = []
    temp_data = []
    for i in range(0, 20):
        angle = random.randint(0, 360)
        rand_hammer = random.randint(0, 4)
        image_hammer_data = transparent_hammers[rand_hammer]
        image_hammer = imutils.rotate(image_hammer_data, angle=angle)
        hammer = Image.fromarray(image_hammer)
        image_width, image_height = hammer.size
        res, mask_binary = cv2.threshold(image_hammer, 1, 255, cv2.THRESH_BINARY)
        mask_binary_image = Image.fromarray(mask_binary.astype(np.uint8))
        mask = np.zeros(shape=(image_dog.height, image_dog.width, 3))
        mask_image = Image.fromarray(mask.astype(np.uint8))
        rand_x = random.randint(0, image_dog.width - image_width)
        rand_y = random.randint(0, image_dog.height - image_height)
        image_dog_copy = image_dog.copy()
        image_dog_copy.paste(hammer, (rand_x, rand_y), hammer)
        image_dog_resized = image_dog_copy.resize((256, 256))
        image_dog_data = np.asarray(image_dog_resized).astype(np.uint8)
        image_dog_data = cv2.cvtColor(image_dog_data, cv2.COLOR_BGR2GRAY) / 255
        mask_image.paste(mask_binary_image, (rand_x, rand_y), mask_binary_image)
        mask_image_resized = mask_image.resize((256, 256))
        mask_image_data = np.asarray(mask_image_resized).astype(np.uint8)
        mask_image_data = cv2.cvtColor(mask_image_data, cv2.COLOR_BGR2GRAY)
        res, mask_image_data = cv2.threshold(mask_image_data, 127, 255, cv2.THRESH_BINARY)
        mask_image_data = mask_image_data/255
        temp_data.append(image_dog_data)
        temp_mask.append(mask_image_data)
        # gdy dane zdjęcia zostały wygenerowane, to dzielę je losowo na zbiory test, train i val
    for j in range(0, 2):
        rand_image = random.randint(0, 19 - (20 - len(temp_data)))
        test_dataset.append(temp_data.pop(rand_image))
        test_mask.append(temp_mask.pop(rand_image))
    for j in range(0, 2):
        rand_image = random.randint(0, 19 - (20 - len(temp_data)))
        val_dataset.append(temp_data.pop(rand_image))
        val_mask.append(temp_mask.pop(rand_image))
    for i in range(0, 16):
        train_dataset.append(temp_data[i])
        train_mask.append(temp_mask[i])


# przerabiam listy na tablice
train_dataset_np = np.array(train_dataset)
train_mask_np = np.array(train_mask)
test_dataset_np = np.array(test_dataset)
test_mask_np = np.array(test_mask)
val_dataset_np = np.array(val_dataset)
val_mask_np = np.array(val_mask)

# muszę dodać jeden wymiar aby móc podawać zdjęcia do sieci
train_dataset_np = np.expand_dims(train_dataset_np, axis=3)
test_dataset_np = np.expand_dims(test_dataset_np, axis=3)
val_dataset_np = np.expand_dims(val_dataset_np, axis=3)
train_mask_np = np.expand_dims(train_mask_np, axis=3)
test_mask_np = np.expand_dims(test_mask_np, axis=3)
val_mask_np = np.expand_dims(val_mask_np, axis=3)

train_mask_np = to_categorical(train_mask_np, num_classes=2)
test_mask_np = to_categorical(test_mask_np, num_classes=2)
val_mask_np = to_categorical(val_mask_np, num_classes=2)

# generuje model
def get_model():
    return multi_unet_model(n_classes=2, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1)

model = get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_dataset_np, train_mask_np,
                    batch_size=16,
                    verbose=1,
                    epochs=10,
                    validation_data=(val_dataset_np, val_mask_np),
                    shuffle=False)

# evaluacja modelu na zbiorze testowym
_, acc = model.evaluate(test_dataset_np, test_mask_np)
print("Accuracy is = ", (acc * 100.0), "%")

# podawanie swoich zdjęć do testowania sieci

link = "http://www.freakingnews.com/pictures/13000/Hammer-13208.jpg"
urllib.request.urlretrieve(link, "image_test.png")
image_test = cv2.imread("image_test.png")
# przygotowanie zdjęcia do podania go na sieć
image_test_pil = Image.fromarray(image_test)
image_resized = image_test_pil.resize((256, 256))
image_resized_cv = np.asarray(image_resized).astype(np.uint8)
image_gray = cv2.cvtColor(image_resized_cv, cv2.COLOR_BGR2GRAY)/255
image_ready = np.array([np.expand_dims(image_gray, axis=2)])

pred_test = model.predict(image_ready)
pred_test_argmax = np.argmax(pred_test, axis=3)

# sprawdzenie maski wygenerowanej z sieci z podanym zdjęciem
image = np.zeros(shape=(256, 256))
for i in range(0, 256):
    for j in range(0, 256):
        if pred_test[0, i, j, 0] == 0:
            image[i, j] = 0
        else:
            image[i, j] = pred_test[0, i, j, 0]

cv2.imshow('img', image)
cv2.waitKey()






