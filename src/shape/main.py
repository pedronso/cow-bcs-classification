from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os


class BcsPolynomialFit:

    def __init__(self):
        # constant parameters
        self.__kernel_size = (3, 3)
        self.__threshold = 10
        self.__polynomial_degree = 30

        self.__characteristic_bcs_info = {}
        self.__characteristic_bcs_images = {}

    def set_characteristic_bcs_images(self, bcs_images: dict):
        self.__characteristic_bcs_images = bcs_images

    def create_characteristic_polynomials(self):
        for bcs, image_path in self.__characteristic_bcs_images.items():
            cow_image, thresh, top_back_shape, x, y, polynomial_coefficients, polynomial = self.__create_polynomial(
                image_path)
            self.__characteristic_bcs_info[bcs] = {
                "image": cow_image,
                "thresh": thresh,
                "top_back_shape": top_back_shape,
                "x": x,
                "y": y,
                "polynomial_coefficients": polynomial_coefficients,
                "polynomial": np.poly1d(polynomial_coefficients)
            }

    def predict(self, image_path: str, real_bcs: float) -> float:
        mse_scores = {}
        cow_image, thresh, top_back_shape, x, y, polynomial_coefficients, polynomial = self.__create_polynomial(
            image_path)

        i = 0
        j = 0
        fig, ax = plt.subplots(2, 4, figsize=(15, 10))

        for bcs, info in self.__characteristic_bcs_info.items():
            new_x_real = np.linspace(sorted(info["x"])[0], sorted(info["x"])[-1], 500)
            new_y_real = info["polynomial"](new_x_real)

            new_x_predict = np.linspace(sorted(x)[0], sorted(x)[-1], 500)
            new_y_predict = polynomial(new_x_predict)

            mse_scores[bcs] = mean_squared_error(new_y_real, new_y_predict)

            if j > 3:
                i += 1
                j = 0

            ax[i, j].plot(new_x_real, new_y_real, "o", markersize=2, color="orange")
            ax[i, j].plot(new_x_predict, new_y_predict, "o", markersize=2)
            ax[i, j].legend([f"known poly - ECC: {bcs}", f"poly to be predicted - ECC: {real_bcs}"], loc="best")

            j += 1

        print(mse_scores)
        plt.show()

        return float(min(mse_scores, key=mse_scores.get))

    def show_characteristic_polynomials(self):
        for bcs, info in self.__characteristic_bcs_info.items():
            fig, ax = plt.subplots(1, 5, figsize=(15, 5))

            ax[0].imshow(cv2.cvtColor(info["image"], cv2.COLOR_GRAY2RGB))
            ax[0].set_title(f"Cow BCS = {bcs}")

            ax[1].imshow(cv2.cvtColor(info["thresh"], cv2.COLOR_GRAY2RGB))
            ax[1].set_title("Thresh image")

            ax[2].imshow(cv2.cvtColor(info["top_back_shape"], cv2.COLOR_GRAY2RGB))
            ax[2].set_title("Contour")

            polynomial_results = info["polynomial"](info["x"])

            ax[3].plot(info["x"], info["y"], "o", markersize=2, color="orange")
            ax[3].plot(info["x"], polynomial_results, "o", markersize=3)
            ax[3].set_title(f"Polynomial degree = {self.__polynomial_degree}")
            ax[3].legend(["cow points", "polynomial"], loc="best")

            new_x_test = np.linspace(sorted(info["x"])[0], sorted(info["x"])[-1], 500)
            new_y_test = info["polynomial"](new_x_test)

            ax[4].plot(new_x_test, new_y_test, "o", markersize=3, color="orange")
            ax[4].plot(info["x"], polynomial_results, "o", markersize=3)
            ax[4].set_title("Resample")
            ax[4].legend(["resampled", "polynomial"], loc="best")

            # break
        plt.show()

    def __create_polynomial(self, image_path: str):
        cow_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(cow_image, self.__threshold, 255, cv2.THRESH_BINARY)

        blur_image = cv2.blur(thresh, self.__kernel_size)
        blur_image[blur_image != 255] = 0

        # dilate the image to prevent black pixels in the center of the cow
        kernel = np.ones(self.__kernel_size, np.uint8)
        dilated_image = cv2.dilate(blur_image, kernel, iterations=2)

        erode_image = self.__erode(dilated_image, kernel_size=self.__kernel_size)
        subtract_image = cv2.subtract(dilated_image, erode_image)
        top_back_shape = self.__get_top_back_shape(subtract_image)

        x, y = self.__translate_shape_coords_to_origin(top_back_shape)
        polynomial_coefficients = np.polyfit(x, y, deg=self.__polynomial_degree)
        polynomial = np.poly1d(polynomial_coefficients)

        return cow_image, thresh, top_back_shape, x, y, polynomial_coefficients, polynomial

    def __erode(self, image, kernel_size=(5, 5), iterations=2):
        kernel = np.ones(kernel_size, np.uint8)
        erode_image = cv2.erode(image, kernel, iterations=iterations)

        return erode_image

    def __get_top_back_shape(self, image):
        image = image.copy()
        mean = np.mean(np.where(image != 0)[0])  # mean of the y coordinates of the board
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if y > mean:
                    image[y][x] = 0

        return image

    def __find_the_center_pixel(self, image):
        mean_y = np.mean(np.where(image != 0)[0])
        mean_x = np.mean(np.where(image != 0)[1])

        return int(mean_x), int(mean_y)

    def __translate_shape_coords_to_origin(self, image):
        flipped_image = image[::-1, :]
        x_distance, y_distance = self.__find_the_center_pixel(flipped_image)

        trans_y = np.where(flipped_image != 0)[0] - y_distance
        trans_x = np.where(flipped_image != 0)[1] - x_distance

        return trans_x, trans_y


def main():
    images_path = os.path.abspath('../../images/grabcut')

    bcs_polynomial_fit = BcsPolynomialFit()
    train_images = {
        2.75: images_path + os.sep + "ECC_2.75" + os.sep + "grabcut_2.png",
        3.0: images_path + os.sep + "ECC_3.0" + os.sep + "grabcut_1.png",
        3.25: images_path + os.sep + "ECC_3.25" + os.sep + "grabcut_1.png",
        3.5: images_path + os.sep + "ECC_3.5" + os.sep + "grabcut_1.png",
        3.75: images_path + os.sep + "ECC_3.75" + os.sep + "grabcut_1.png",
        4.0: images_path + os.sep + "ECC_4.0" + os.sep + "grabcut_4.png",
        4.5: images_path + os.sep + "ECC_4.5" + os.sep + "grabcut_1.png",
    }
    bcs_polynomial_fit.set_characteristic_bcs_images(train_images)
    bcs_polynomial_fit.create_characteristic_polynomials()
    # bcs_polynomial_fit.show_characteristic_polynomials()

    results = {"right": 0, "wrong": 0}

    # print(f'The predicted bcs is: {bcs_polynomial_fit.predict(images_path + os.sep + "ECC_3.0" + os.sep + "grabcut_4.png", 3.0)}')

    # directory[0] -> directory path
    # directory[1] -> subdirectories names
    # directory[2] -> directory files
    for directory in os.walk(images_path):
        if len(directory[1]) == 0:
            for image_file in directory[2]:  # walk through the image files in directories
                test_cow = directory[0] + os.sep + image_file
                real_bcs = float(
                    directory[0].split(os.sep)[-1].split("_")[-1])  # get the BCS number from the directory name

                if train_images[real_bcs] != test_cow:  # check if the test image is the train image
                    predicted_bcs = bcs_polynomial_fit.predict(test_cow, real_bcs)
                    if real_bcs == predicted_bcs:
                        results["right"] += 1
                    else:
                        results["wrong"] += 1

    print(results)


if __name__ == "__main__":
    main()
