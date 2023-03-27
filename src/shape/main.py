import cv2
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

    def mean_squared(self,x_real, y_real, x_predict, y_predict, points):
        poly1= np.array([list(a) for a in zip(x_real,y_real)])
        poly2= np.array([list(a) for a in zip(x_predict,y_predict)])

        return (np.sum([np.linalg.norm(p1 - p2)**2 for p1, p2 in zip(poly1, poly2)])/points)

    def predict(self, image_path: str, real_bcs: float) -> float:
        points = 500
        mse_scores = {}
        cow_image, thresh, top_back_shape, x, y, polynomial_coefficients, polynomial = self.__create_polynomial(
            image_path)

        i = 0
        j = 0
        fig, ax = plt.subplots(2, 4, figsize=(15, 10))

        for bcs, info in self.__characteristic_bcs_info.items():
            min_x_real = sorted(info["x"])[0]
            #print(bcs,"min_X_real: ",min_x_real)
            max_x_real = sorted(info["x"])[-1]
            #print(bcs,"max_X_real: ",max_x_real)
            min_x_predict = sorted(x)[0]
            #print(bcs,"min_X_predict: ",min_x_predict)
            max_x_predict = sorted(x)[-1]
            #print(bcs,"max_X_predict: ",max_x_predict)

            new_x_real = np.linspace(min_x_real, max_x_real, points)
            #print(bcs,"new_X_real: ",new_x_real)
            new_y_real = info["polynomial"](new_x_real)
            #print(bcs,"new_Y_real: ",new_y_real)
            new_x_predict = np.linspace(min_x_predict, max_x_predict, points)
            #print(bcs,"new_X_predict: ",new_x_predict)
            new_y_predict = polynomial(new_x_predict)
            #print(bcs,"new_Y_predict: ",new_y_predict)

            min_y_real = sorted(new_y_real)[0]
            max_y_real = sorted(new_y_real)[-1]
            min_y_predict = sorted(new_y_predict)[0]
            max_y_predict = sorted(new_y_predict)[-1]

            cx = (max_x_predict - min_x_predict) / (max_x_real - min_x_real)
            cy = (max_y_predict - min_y_predict) / (max_y_real - min_y_real)
            mse_scores[bcs] = mean_squared_error(new_y_real, new_y_predict)

            #print(cx," = ",max_x_predict, "-", min_x_predict ,"/", max_x_real,"-", min_x_real)
            #print(bcs,"new_X_predict: ",new_x_predict/cx)
            #print(cy," = ",max_y_predict, "-", min_y_predict ,"/", max_y_real,"-", min_y_real)
            #print(bcs,"new_Y_predict: ",new_y_predict/cy)

            #print(new_y_predict)
            #print("mse: ",new_y_real,",",new_y_predict/cy)
            #print(len(new_y_predict),len(new_y_real))
        
            #mse normal só com y:
            #distância entre pontos com x e y:
            #mse_scores[bcs] = self.mean_squared(new_x_real,new_y_real, new_x_predict/cx, new_y_predict / cy, points)
            #print(mse_scores[bcs])

            if j > 3:
                i += 1
                j = 0

            ax[i, j].plot(new_x_real, new_y_real, "o", markersize=2, color="orange")
            ax[i, j].plot(new_x_predict , new_y_predict , "o", markersize=2, color="green")
            ax[i, j].axis("equal")
            ax[i, j].legend([f"known poly - ECC: {bcs}", f"resized poly - ECC: {real_bcs}"], loc="best")

            j += 1

        #print(polynomial)
        print(image_path)
        print(mse_scores)
        plt.show()

        return float(min(mse_scores, key=mse_scores.get))

    def derivative_analysis(self):
        i = 0
        j = 0
        fig, ax = plt.subplots(2, 4, figsize=(15, 10))

        for bcs, info in self.__characteristic_bcs_info.items():
            min_x_real = sorted(info["x"])[0]
            max_x_real = sorted(info["x"])[-1]

            new_x_real = np.linspace(min_x_real, max_x_real, 500)
            new_y_real = info["polynomial"](new_x_real)

            if j > 3:
                i += 1
                j = 0

            real_poly_derivative = np.gradient(info["polynomial"](new_x_real), new_x_real)

            critical_points = new_x_real[1:][(real_poly_derivative[1:] * real_poly_derivative[:-1]) <= 0]
            print(f"ECC: {bcs} - Qtd. de pontos críticos: {len(critical_points)}")

            ax[i, j].plot(new_x_real, new_y_real, "o", markersize=2, color="blue")
            ax[i, j].plot(new_x_real, real_poly_derivative, "o", markersize=2, color="green")
            ax[i, j].plot(critical_points, info["polynomial"](critical_points), "o", markersize=5, color="red")
            ax[i, j].legend([f"ECC: {bcs}", "derivative", "critical points"], loc="upper right")

            ax[i, j].axis("equal")

            j += 1
        plt.show()

    def show_characteristic_polynomials(self):
        for bcs, info in self.__characteristic_bcs_info.items():
            fig, ax = plt.subplots(1, 5, figsize=(12, 5))

            ax[0].imshow(cv2.cvtColor(info["image"], cv2.COLOR_GRAY2RGB))
            ax[0].set_title(f"Cow BCS = {bcs}")

            ax[1].imshow(cv2.cvtColor(info["thresh"], cv2.COLOR_GRAY2RGB))
            ax[1].set_title("Thresh image")

            ax[2].imshow(cv2.cvtColor(info["top_back_shape"], cv2.COLOR_GRAY2RGB))
            ax[2].set_title("Contour")

            polynomial_results = info["polynomial"](info["x"])

            ax[3].plot(info["x"], info["y"], "o", markersize=2, color="orange")
            ax[3].set_title(f"Cow countour")
            ax[3].legend(["cow points"], loc="best")
            ax[3].axis("equal")

            ax[4].plot(info["x"], polynomial_results, "o", markersize=3)
            ax[4].set_title(f"Polynomial degree = {self.__polynomial_degree}")
            ax[4].legend(["polynomial"], loc="best")
            ax[4].axis("equal")
            

            # break
        plt.show()

    def __create_polynomial(self, image_path: str):
        cow_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # cv2.imshow("img",cow_image)
        # cv2.imwrite("img.jpg",cow_image)
        
        _, thresh = cv2.threshold(cow_image, self.__threshold, 255, cv2.THRESH_BINARY)
        # cv2.imshow("img threshold",thresh)
        # cv2.imwrite("img_threshold.jpg",thresh)


        blur_image = cv2.blur(thresh, self.__kernel_size)
        # cv2.imshow("img blur",blur_image)
        # cv2.imwrite("img_blur.jpg",blur_image)
        

        blur_image[blur_image != 255] = 0
        # cv2.imshow("img blur pos",blur_image)
        # cv2.imwrite("img blur pos",blur_image)
        

        # dilate the image to prevent black pixels in the center of the cow
        kernel = np.ones(self.__kernel_size, np.uint8)
        # print("img kernel",kernel)

        dilated_image = cv2.dilate(blur_image, kernel, iterations=2)
        # cv2.imshow("img dilated",dilated_image)
        # cv2.imwrite("img_dilated.jpg",dilated_image)

        erode_image = self.__erode(dilated_image, kernel_size=self.__kernel_size)
        # cv2.imshow("img erode",erode_image)
        # cv2.imwrite("img_erode.jpg",erode_image)

        subtract_image = cv2.subtract(dilated_image, erode_image)
        # cv2.imshow("img subtract",subtract_image)
        # cv2.imwrite("img_subtract.jpg",subtract_image)

        top_back_shape = self.__get_top_back_shape(subtract_image)
        # cv2.imshow("img top",top_back_shape)
        # cv2.imwrite("img_top.jpg",top_back_shape)

        cv2.waitKey(0)
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

    images_path = r'images\resized_images_error_test\all_images'

    bcs_polynomial_fit = BcsPolynomialFit()
    train_images = {
        2.50: images_path + os.sep + "2.50" + os.sep + "3.jpeg",
        2.75: images_path + os.sep + "2.75" + os.sep + "2.jpeg",
        3.0: images_path + os.sep + "3.0" + os.sep + "4.jpeg",
        3.25: images_path + os.sep + "3.25" + os.sep + "2.jpeg",
        3.50: images_path + os.sep + "3.50" + os.sep + "18.jpeg",
        3.75: images_path + os.sep + "3.75" + os.sep + "3.jpeg",
        4.0: images_path + os.sep + "4.0" + os.sep + "7.jpeg"
    }


    bcs_polynomial_fit.set_characteristic_bcs_images(train_images)
    bcs_polynomial_fit.create_characteristic_polynomials()
    # bcs_polynomial_fit.show_characteristic_polynomials()
    #bcs_polynomial_fit.derivative_analysis()

    results = {"right": 0, "wrong": 0}

    # print(f'The predicted bcs is: {bcs_polynomial_fit.predict(images_path + os.sep + "ECC_3.0" + os.sep + "grabcut_2.png", 3.25)}')

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
                    print(f"Real: {real_bcs} - Predicted: {predicted_bcs}")
                    if real_bcs == predicted_bcs:
                        results["right"] += 1
                    else:
                        results["wrong"] += 1
    

    print(results)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()

