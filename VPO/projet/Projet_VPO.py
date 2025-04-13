import math
from random import randint
from skimage.morphology import flood
import tkinter as tk
from tkinter import filedialog ,simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import customtkinter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import segmentation, color
from tkinter import messagebox

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.resizable(False, False)
        self.original_image=None
        self.modified_image=None
        self.filtre_kernel=None
        self.filter_haut_bas=""
        self.type_filtre = ""
        self.taille_filtre_rows = 0
        self.taille_filtre_cols = 0
        self.cutoff_frequency = 0.0
        # configure window
        self.title("Projet VPO")
        self.geometry(f"{1474}x{780}")
        #self.resizable(width=False,height=False)

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=1)  # Ajustement pour la colonne 3
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(13, weight=1)  # Correction de l'index
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="VPO Project", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(10, 8))
        self.download_button = customtkinter.CTkButton(self.sidebar_frame, text="Insérer Image", command=self.open_image)
        self.download_button.grid(row=1, column=0, padx=20, pady=(20, 10))

        self.menu_1 = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Transformation", "Negative", "Rotation", "Redimension", "Rectangle", "Histogramme NG", "Histogramme RGB", "Etirement", "Egalisation"],command=self.menu_selected)
        self.menu_1.grid(row=2, column=0, padx=20, pady=10)
        self.menu_2 = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Binarisation", "seuillage manuel", "OTSU", "Moyenne ponderée"], command=self.menu_selected)
        self.menu_2.grid(row=3, column=0, padx=20, pady=10)
        self.menu_3 = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Filtrage", "Gaussien", "Moyenneur", "Median"], command=self.menu_selected)
        self.menu_3.grid(row=4, column=0, padx=20, pady=10)
        self.menu_9 = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Filtrage fréquentiel", "Passe Bas","Passe Haut"], command=self.menu_selected)
        self.menu_9.grid(row=5, column=0, padx=20, pady=10)
        self.menu_4 = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Extraction contour", "Gradient", "Sobel", "Robert" ,"Laplacien"], command=self.menu_selected)
        self.menu_4.grid(row=6, column=0, padx=20, pady=10)
        self.menu_5 = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Morphologie", "Erosion", "Dilatation" , "Ouverture" , " Fermeture" , "Filtrage Morphologique"], command=self.menu_selected)
        self.menu_5.grid(row=7, column=0, padx=20, pady=10)
        self.menu_6 = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Segmentation", "Croissance de regions", "Partition de regions" , "k means","k means color"], command=self.menu_selected)
        self.menu_6.grid(row=8, column=0, padx=20, pady=10)
        self.menu_7 = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Point d'interet", "hough", "Hariss","sift"], command=self.menu_selected)
        self.menu_7.grid(row=9, column=0, padx=20, pady=10)
        self.menu_8 = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Compression", "Huffman","LZW","Ondelette"], command=self.menu_selected)
        self.menu_8.grid(row=10, column=0, padx=20, pady=10)
        
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Thème:", anchor="w")
        self.appearance_mode_label.grid(row=11, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=12, column=0, padx=20, pady=(10, 10))


        frame_original = customtkinter.CTkFrame(master=self, width=620, height=620, corner_radius=10)
        frame_original.grid(row=0, column=1, padx=5, pady=5)

        frame_modified = customtkinter.CTkFrame(master=self, width=620, height=620, corner_radius=10)
        frame_modified.grid(row=0, column=2, padx=5, pady=5)

        # Créer les titres
        title_original = customtkinter.CTkLabel(master=frame_original, text="Image Originale", font=customtkinter.CTkFont(size=14, weight="bold"))
        title_original.pack(pady=10)

        title_modified = customtkinter.CTkLabel(master=frame_modified, text="Image Modifiée", font=customtkinter.CTkFont(size=14, weight="bold"))
        title_modified.pack(pady=10)
        
        # Création des deux zones d'affichage pour les images
        # Créer les canvas à l'intérieur des cadres
        self.canvas_original_image = tk.Canvas(frame_original, width=600, height=600)
        self.canvas_original_image.pack()

        self.canvas_modified_image = tk.Canvas(frame_modified, width=600, height=600)
        self.canvas_modified_image.pack()

        # Ajout du bouton "Enregistrer Résultat" au centre et en dessous des cadres
        button_frame = customtkinter.CTkFrame(master=self, width=620, height=50)
        button_frame.grid(row=1, column=1, columnspan=2)  # Span sur 2 colonnes

        save_button = customtkinter.CTkButton(master=button_frame, text="Enregistrer Résultat",command=self.save_result)
        save_button.place(relx=0.5, rely=0.5, anchor="center")  # Centrer le bouton dans le cadre

    def save_result(self):
        if self.modified_image is None:
            messagebox.showerror("Erreur", "Veuillez Appliquer une Operation!")
        else:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")))
            if file_path:
                image = Image.fromarray(self.modified_image)
                image.save(file_path)
                print("Image sauvegardée avec succès sous:", file_path)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)
        # Initialize original and modified images
        self.original_image = None
        self.modified_image = None
    def afficher_filtre(self):
        dialog = FiltreDialog(self,titre=self.filter_haut_bas)
        result=dialog.result
        rows,cols,_=self.original_image.shape
        # Création du filtre en fonction du type choisi
        if result["type_filtre"] == "butterworth":
            if self.filter_haut_bas=="bas":
                self.filtre_kernel = self.butterworth_lowpass_filter(rows,cols, int(result["cutoff_frequency"]),int(result["butterworth_order"]))
            else:       
                self.filtre_kernel = self.butterworth_highpass_filter(rows,cols, int(result["cutoff_frequency"]),int(result["butterworth_order"]))
        elif result["type_filtre"] == "gaussien":
            if self.filter_haut_bas=="bas":
                self.filtre_kernel = self.gaussian_lowpass_filter(rows,cols, int(result["cutoff_frequency"]))
            else:       
                self.filtre_kernel = self.gaussian_highpass_filter(rows,cols, int(result["cutoff_frequency"]))
        elif result["type_filtre"] == "ideal":
            if self.filter_haut_bas=="bas":
                self.filtre_kernel = self.ideal_lowpass_filter(rows,cols, int(result["cutoff_frequency"]))
            else:       
                self.filtre_kernel = self.ideal_highpass_filter(rows,cols, int(result["cutoff_frequency"]))

        # Appliquer le filtre à l'image originale
        
        self.modified_image = self.application_filtre()
        self.display_modified_image()

    def ideal_lowpass_filter(self,rows, cols, cutoff_freq):
        v = np.arange(cols)
        u = np.arange(rows)
        V, U = np.meshgrid(v, u)  # Inversion de l'ordre des dimensions
        D = np.sqrt((U - rows / 2)**2 + (V - cols / 2)**2)
        H = np.where(D <= cutoff_freq, 1, 0)
        return H

    def ideal_highpass_filter(self,rows, cols, cutoff_freq):
        return 1 - self.ideal_lowpass_filter(rows, cols, cutoff_freq)
    
    def gaussian_lowpass_filter(self,rows, cols, cutoff_freq):
        u = np.arange(rows)
        v = np.arange(cols)
        V,U = np.meshgrid(v,u)
        D = np.sqrt((U - rows / 2)**2 + (V - cols / 2)**2)
        H = np.exp(-(D**2) / (2 * cutoff_freq**2))
        return H

    def gaussian_highpass_filter(self,rows, cols, cutoff_freq):
        return 1 - self.gaussian_lowpass_filter(rows, cols, cutoff_freq)

    def butterworth_lowpass_filter(self,rows, cols, cutoff_freq, n):
        u = np.arange(rows)
        v = np.arange(cols)
        V,U= np.meshgrid(v,u)
        D = np.sqrt((U - rows / 2)**2 + (V - cols / 2)**2)
        H = 1 / (1 + (D / cutoff_freq)**(2 * n))
        return H

    def butterworth_highpass_filter(self,rows, cols, cutoff_freq, n):
        return 1 - self.butterworth_lowpass_filter(rows, cols, cutoff_freq, n)
    
    def apply_filter(self, image):
        spectrum = np.fft.fft2(image)
        spectrum_shifted = np.fft.fftshift(spectrum)
        filtered_spectrum = spectrum_shifted * self.filtre_kernel
        filtered_spectrum_shifted = np.fft.ifftshift(filtered_spectrum)
        filtered_image = np.fft.ifft2(filtered_spectrum_shifted)
        filtered_image = np.abs(filtered_image)
        return filtered_image

    def application_filtre(self):
        # Vérifier si l'image est en couleur (RGB)
        if len(self.original_image.shape) == 3:
            # Appliquer le filtre sur chaque canal séparément
            filtered_images = []
            for channel in range(self.original_image.shape[2]):
                filtered_image_channel = self.apply_filter(self.original_image[:, :, channel])
                filtered_images.append(filtered_image_channel)
            # Fusionner les canaux filtrés en une seule image RGB
            filtered_image_rgb = np.stack(filtered_images, axis=2)
            filtered_image_rgb = cv2.convertScaleAbs(filtered_image_rgb)
            return filtered_image_rgb
        else:
            # Si l'image n'est pas en couleur (RGB), retourner None ou prendre une autre action appropriée
            return None


    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)
    
    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)
        # Initialize original and modified images
        self.original_image = None
        self.modified_image = None

    def open_image(self):
        imgPath = filedialog.askopenfilename()
        if imgPath:
            self.original_image = cv2.imread(imgPath)
            self.display_original_image()

    def display_original_image(self):
        if self.original_image is not None:
            image = self.original_image
            image = cv2.resize(np.copy(image), (600, 600))
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image_pil)
            self.canvas_original_image.delete("all")
            self.canvas_original_image.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas_original_image.image = photo

    def display_modified_image(self):
        self.resized_img = None
        if self.modified_image is not None:
            image = self.modified_image
            image = cv2.resize(np.copy(image), (600, 600))
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image_pil)
            self.canvas_modified_image.delete("all")
            self.canvas_modified_image.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas_modified_image.image = photo


    

    def erosion(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2GRAY)
            ES = np.ones((5,5), np.uint8)
            eroded = cv2.erode(src=gray, kernel=ES, iterations=1)
            self.modified_image = eroded
            self.display_modified_image()


    def dilatation(self) :
        if self.original_image is not None :
            gray = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2GRAY)
            ES= np.ones((5,5),np.uint8)
            dilated = cv2.dilate(src=gray, kernel=ES, iterations=1)
            self.modified_image = cv2.cvtColor(dilated, cv2.COLOR_GRAY2RGB)
            self.display_modified_image()


    def ouverture(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2GRAY)
            ES = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
            opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, ES)
            self.modified_image = opening
            self.display_modified_image()


    def fermeture(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2GRAY)
            ES = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
            img = np.array(gray)
            self.modified_image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, ES)
            self.display_modified_image()


    def negative(self):
        if self.original_image is not None:
            img = np.array(self.original_image)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Appliquer le négatif
            img_negative = 255 - img

            self.modified_image = img_negative
            self.display_modified_image()


    def etirement(self, lower_pct=5, upper_pct=95):
        if self.original_image is not None:
            img = np.array(self.original_image)
            if len(img.shape) == 3: # pour les images couleur
                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            else: # pour les images en niveaux de gris
                p_lower, p_upper = np.percentile(img, (lower_pct, upper_pct))
                img_etiree = (img - p_lower) * (255 / (p_upper - p_lower))
                img_etiree = np.clip(img_etiree, 0, 255).astype(np.uint8)
                img = img_etiree
            
            self.modified_image = img
            self.display_modified_image()



    def egalisation(self):
        if self.original_image is not None:
            img = np.array(self.original_image)
            if len(img.shape) == 3:  # Si l'image est en couleur
                b, g, r = cv2.split(img)

                b_eq = cv2.equalizeHist(b)
                g_eq = cv2.equalizeHist(g)
                r_eq = cv2.equalizeHist(r)

                img_equalized = cv2.merge((b_eq, g_eq, r_eq))
            else:  # Si l'image est en niveaux de gris
                img_equalized = cv2.equalizeHist(img)

            self.modified_image = img_equalized
            self.display_modified_image()


    def histogramme(self):
        if self.original_image is not None:
            img = np.array(self.original_image)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                plt.plot(hist, color='black')
                plt.xlabel('Niveau de gris')
                plt.ylabel('Nombre de pixels')
                plt.show()

    
    def histogrammeRGB(self):
        if self.original_image is not None:
            img = np.array(self.original_image)
            colors = ('b', 'g', 'r')
            plt.figure()
            for i, col in enumerate(colors):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
            plt.xlim([0, 256])
            plt.xlabel('Couleur RGB')
            plt.ylabel('Nombre de pixels')
            plt.show()


    def rotation(self):
        if self.original_image is not None:
            angle = simpledialog.askfloat("Rotation", "Entrez l'angle de rotation en degrés : ",
                                           parent=self.master)
            if angle is None:
                return
            rows, cols = self.original_image.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            self.modified_image = cv2.warpAffine(self.original_image, M, (cols, rows))
            self.display_modified_image()


    def sobel(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sx = cv2.convertScaleAbs(sobel_x)
            sy = cv2.convertScaleAbs(sobel_y)
            self.modified_image = cv2.addWeighted(sx, 0.5, sy, 0.5, 0)
            self.display_modified_image()


    def gradient(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0 ,ksize=3)
            dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1 ,ksize=3)
            magnitude = np.sqrt(dx**2 + dy**2)
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            magnitude = np.uint8(magnitude)
            _, thresholded = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
            self.modified_image = thresholded
            self.display_modified_image()


    def robert(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2GRAY)
            roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
            roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
            gradient_x = cv2.filter2D(gray, cv2.CV_32F, roberts_x)
            gradient_y = cv2.filter2D(gray, cv2.CV_32F, roberts_y)

            # Calculez le module du gradient
            magnitude = cv2.magnitude(gradient_x, gradient_y)
            seuil = 50
            _, seuil_img = cv2.threshold(magnitude, seuil, 255, cv2.THRESH_BINARY)
            self.modified_image =seuil_img
            self.display_robert_image()

    
    def Hough(self):
        image=self.original_image
        img = np.array(image)
        # Convertir l'image en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Appliquer un filtre de détection de contours
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # Détecter les cercles avec la méthode de Hough
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0,
                                maxRadius=0)
        # Dessiner les cercles détectés sur l'image d'entrée
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        self.modified_image= img
        self.display_modified_image()


    def display_robert_image(self):
        if self.modified_image is not None:
            # Convertir le tableau numpy en uint8 avant de le convertir en image PIL
            image = cv2.resize(np.copy(self.modified_image), (600, 600))
            image_pil = Image.fromarray(cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2RGB))
            self.image_tk_modified = ImageTk.PhotoImage(image_pil)
            photo = self.image_tk_modified
            self.canvas_modified_image.delete("all")
            self.canvas_modified_image.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas_modified_image.image = photo


    

    def laplacien(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = cv2.convertScaleAbs(laplacian)
            seuil = 30
            _, seuil_img = cv2.threshold(laplacian, seuil, 255, cv2.THRESH_BINARY)
            self.modified_image = seuil_img
            self.display_modified_image()

    def filtrage_morphologique(self):
        if self.original_image is not None:
            gray = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2GRAY)
            kernel = np.ones((3,3),np.uint8)
            morph = cv2.morphologyEx(gray,cv2.MORPH_OPEN, kernel)
            self.modified_image = morph
            self.display_modified_image()



    def partitionD(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marquer les régions inconnues comme 0
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Appliquer l'algorithme Watershed
        markers = cv2.watershed(self.original_image, markers)

        # Créer une image de sortie marquant les bords avec une couleur rouge
        image_out = np.zeros_like(self.original_image)
        image_out[markers == -1] = [255, 0, 0]

        self.modified_image = image_out
        self.display_modified_image()



    def regionGrowing(self):
        img=self.original_image
        tolerance = 10

        img = img

        new_image = img.copy()
        img_gray = img.copy()
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.medianBlur(img_gray, 9)
        seed = (0, 0)
        new_value = img[seed]
        all_visited = False
        visited = np.zeros(img_gray.shape)

        while visited.min() == 0:
            mask = flood(img_gray, seed, tolerance=tolerance)
            visited[mask] = 1
            new_image[mask] = new_value
            not_visited = np.where(visited == 0)
            all_visited = len(not_visited[0]) == 0
            if not all_visited:
                seed = (not_visited[0][0], not_visited[1][0])
                new_value = img[seed]
        self.modified_image= new_image
        self.display_modified_image()

    def regionSpliting(image):
        # Convertir l'image en nuance de gris
        gray_image = color.rgb2gray(image)
        # Définir la taille des blocs
        block_size = 50
        # Calculer les dimensions de la grille de blocs
        height, width = gray_image.shape
        num_rows = int(np.ceil(height / block_size))
        num_cols = int(np.ceil(width / block_size))
        # Ajouter des bords pour s'assurer que la grille de blocs est complète
        pad_height = num_rows * block_size - height
        pad_width = num_cols * block_size - width
        padded_image = np.pad(gray_image, ((0, pad_height), (0, pad_width)), mode='constant')
        # Diviser l'image en blocs
        blocks = np.zeros((num_rows, num_cols, block_size, block_size))
        for row in range(num_rows):
            for col in range(num_cols):
                block = padded_image[row * block_size:(row + 1) * block_size, col * block_size:(col + 1) * block_size]
                blocks[row, col] = block
        # Calculer la moyenne de chaque bloc
        block_means = np.mean(blocks, axis=(2, 3))
        # Utiliser l'algorithme de regroupement pour segmenter l'image
        labels = segmentation.slic(image, n_segments=100, compactness=10, sigma=1, start_label=1)
        return segmentation.mark_boundaries(image, labels)

    def KMeansSegmentation(self):
        image=self.original_image
        final_image = np.copy(image)
        Z = final_image.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 8
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((final_image.shape))
        self.modified_image= res2
        self.display_modified_image()

    def kmeans_color(self, k=2, max_iter=100):
        img=self.original_image
        # Convertir l'image en un tableau numpy (hauteur x largeur x 3)
        img_data = np.array(img)
        # Obtenir la hauteur et la largeur de l'image
        h, w = img_data.shape[:2]
        # Initialiser les centres aléatoirement
        centers = np.zeros((k, 3))
        for i in range(k):
            x = randint(0, w - 1)
            y = randint(0, h - 1)
            centers[i] = img_data[y, x]
        # Appliquer l'algorithme K-means
        for i in range(max_iter):
            # Assigner chaque pixel au centre le plus proche
            labels = np.zeros((h, w), dtype=np.int32)
            for y in range(h):
                for x in range(w):
                    distances = np.linalg.norm(centers - img_data[y, x], axis=1)
                    labels[y, x] = np.argmin(distances)
            # Calculer les nouveaux centres
            new_centers = np.zeros((k, 3))
            counts = np.zeros((k,))
            for y in range(h):
                for x in range(w):
                    label = labels[y, x]
                    new_centers[label] += img_data[y, x]
                    counts[label] += 1
            for i in range(k):
                if counts[i] > 0:
                    new_centers[i] /= counts[i]
            # Vérifier si les centres ont convergé
            if np.allclose(centers, new_centers):
                break
            # Mettre à jour les centres
            centers = new_centers
        # Créer une image segmentée en couleur
        segmented_img = np.zeros_like(img_data)
        for y in range(h):
            for x in range(w):
                label = labels[y, x]
                segmented_img[y, x] = centers[label]
        # Retourner l'image segmentée en couleur
        self.modified_image= segmented_img
        self.display_modified_image()


    def reset_image(self):
        if self.original_image is not None:
            self.modified_image = self.original_image.copy()
            self.display_modified_image()


    def selection_image(self):
        self.modified_image = None
        self.resized_img = None
        if self.original_image.size > 0:

            r1 = messagebox.showinfo("Sélection", 'Veuillez sélectionner une région et cliquer sur le bouton "espace" our "entrer" pour voir la région sélectionnée')

            if r1 == "ok":
                roi = cv2.selectROI(self.original_image)
                self.modified_image = self.original_image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
                self.display_modified_image()
        else:
            messagebox.showerror("Erreur", "Veuillez choisir une image!")



    def redimensionner(self):
        self.modified_image = None
        if self.original_image is not None:
            width = simpledialog.askinteger("Redimensionnement", "Entrez la nouvelle largeur : ", parent=self.master)
            height = simpledialog.askinteger("Redimensionnement", "Entrez la nouvelle hauteur : ", parent=self.master)

            img = Image.fromarray(self.original_image)
            self.resized_img = img.resize((width, height))
            self.modified_image=self.resized_img
            self.display_resized_image()

    def display_resized_image(self):        
        self.canvas_modified_image.delete("all")

        if self.resized_img is not None:
            image = np.array(self.modified_image)
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(image_pil)
            self.canvas_modified_image.delete("all")
            self.canvas_modified_image.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas_modified_image.image = photo


    def filter_gaussian(self):
        if self.original_image is not None :
            sigma = simpledialog.askfloat("Filtre Gaussien", "Entrez la valeur de l'écart type (entre 0.5 et 5.0) :"
                                          , parent=self.master)
            self.modified_image = cv2.GaussianBlur(self.original_image, (0, 0), sigmaX=sigma, sigmaY=sigma)
            self.display_modified_image()


    def filter_moyenneur(self):
        if self.original_image is not None:
            filter_size = simpledialog.askinteger("Filtre Moyenneur", "Entrez la taille du filtre" 
                                                  "(nombre impair >3) :", parent=self.master)
            self.modified_image = cv2.blur(self.original_image, (filter_size, filter_size))
            self.display_modified_image()


    def filter_median(self):
        if self.original_image is not None:
            filter_size = simpledialog.askinteger("Filtre Médian", "Entrez la taille du filtre (nombre impair) :"
                                                  , parent=self.master)
            self.modified_image = cv2.medianBlur(self.original_image, filter_size)
            self.display_modified_image()


    def binarize_global(self):
        if self.original_image is not None:
            threshold = simpledialog.askinteger("Seuillage manuel", "Entrez la valeur de seuil (entre 0 et 255) :"
                                                , parent=self.master)
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
            self.modified_image = binary_image
            self.display_modified_image()


    def binarize_otsu(self):
         if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.modified_image = binary_image
            self.display_modified_image()


    def binarize_weighted_mean(self):
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            threshold_value = cv2.mean(gray_image)[0]
            ret, self.modified_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
            self.display_modified_image()


    def compress_huffman(self):
        file = filedialog.asksaveasfile(mode='wb', defaultextension=".jpg")
        if file:
            img_encode = cv2.imencode('.jpg',self.original_image)[1]
            data_encode = np.array(img_encode)
            str_encode = data_encode.tostring()
            file.write(str_encode)
            file.close()

    def compress_lzw(self):
        None

    
    def Hariss(self):
        img = np.array(self.original_image)
        # Convertir l'image en niveaux de gris
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Appliquer un filtre de détection de contours
        gray = np.float32(gray)
        # Détecter les cercles avec la méthode de Hough
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        img[dst > 0.01 * dst.max()] = [255, 0, 0]
        self.modified_image= img
        self.display_modified_image()


    def Sift(self):
        img = np.array(self.original_image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp = sift.detect(gray, None)
        img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.modified_image= img
        self.display_modified_image()

    
    def contourRoberts(self):
        image=self.original_image
        image = np.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Appliquer le filtre de Roberts
        kernelx = np.array([[1, 0], [0, -1]])
        kernely = np.array([[0, 1], [-1, 0]])
        robertsx = cv2.filter2D(gray, -1, kernelx)
        robertsy = cv2.filter2D(gray, -1, kernely)
        cx =cv2.convertScaleAbs(robertsx)
        cy = cv2.convertScaleAbs(robertsy)
        grad = cv2.addWeighted(cx, 0.5, cy, 0.5, 0)
        self.modified_image= grad
        self.display_modified_image()



    def contourLaplacien(self):
        image=self.original_image
        # image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        # Appliquer le filtre de Laplacien
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        cx = cv2.convertScaleAbs(laplacian)
        _, thresh = cv2.threshold(cx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Appliquer une opération de dilatation
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(thresh, kernel, iterations=1)
        # Trouver les contours de l'image résultante
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Dessiner les contours trouvés
        cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
        self.modified_image= image
        self.display_modified_image()


    def contourGradient(self):
        img = self.original_image
        seuil = 128
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        IX = img_gray.copy().astype('int32')
        IY = img_gray.copy().astype('int32')
        for i in range(1, img_gray.shape[0] - 1):
            for j in range(1, img_gray.shape[1] - 1):
                IX[i, j] = int(img_gray[i, j + 1]) - int(img_gray[i, j])
                IY[i, j] = int(img_gray[i + 1, j]) - int(img_gray[i, j])
        IR = img_gray.copy()
        for i in range(1, img_gray.shape[0] - 1):
            for j in range(1, img_gray.shape[1] - 1):
                IR[i, j] = math.sqrt(IX[i, j] ** 2 + IY[i, j] ** 2)
                if IR[i, j] < seuil:
                    IR[i, j] = 0
                else:
                    IR[i, j] = 255
        self.modified_image = IR
        self.display_modified_image()

    def contourSobel(self):
        image=self.original_image
        image = np.array(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Appliquer un filtre de détection de contour
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        cx = cv2.convertScaleAbs(grad_x)
        cy = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(cx, 0.5, cy, 0.5, 0)
        # Appliquer une opération de seuillage
        seuil, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print('Seuil :', seuil)
        # Appliquer une opération de dilatation
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(thresh, kernel, iterations=1)
        # Trouver les contours de l'image résultante
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Dessiner les contours trouvés
        cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
        self.modified_image= image
        self.display_modified_image()

    def menu_selected(self, value):
        if self.original_image is None:
            messagebox.showerror("Erreur", "Veuillez choisir une image!")
        else:
    # Perform image transformation based on selected menu option
            if value == "Negative":
                self.negative() 
            elif value == "Rotation":
                self.rotation()  # À remplir avec votre fonction de rotation
            elif value == "Redimension":
                self.redimensionner()  # À remplir avec votre fonction de redimension
            elif value == "Rectangle":
                self.selection_image()  # À remplir avec votre fonction de dessin de rectangle
            elif value == "Histogramme NG":
                self.histogramme()  # À remplir avec votre fonction de binarisation
            elif value == "Histogramme RGB":
                self.histogrammeRGB()  # À remplir avec votre fonction de seuillage manuel
            elif value == "Etirement":
                self.etirement()  # À remplir avec votre fonction de seuillage OTSU
            elif value == "Egalisation":
                self.egalisation()
            elif value == "Seuillage manuel":
                self.binarize_global()  # À remplir avec votre fonction de rotation
            elif value == "OTSU":
                self.binarize_otsu()  # À remplir avec votre fonction de redimension
            elif value == "Moyenne pondérée":
                self.binarize_weighted_mean()  # À remplir avec votre fonction de dessin de rectangle
            elif value == "Gaussien":
                self.filter_gaussian()  # À remplir avec votre fonction de binarisation
            elif value == "Moyenneur":
                self.filter_moyenneur()  # À remplir avec votre fonction de seuillage manuel
            elif value == "Median":
                self.filter_median()  # À remplir avec votre fonction de seuillage OTSU
            elif value == "Gradient":
                self.contourGradient() 
            elif value == "Sobel":
                self.contourSobel()  # À remplir avec votre fonction de rotation
            elif value == "Robert":
                self.contourRoberts()  # À remplir avec votre fonction de redimension
            elif value == "Laplacien":
                self.contourLaplacien()  # À remplir avec votre fonction de dessin de rectangle
            elif value == "Erosion":
                self.erosion()  # À remplir avec votre fonction de binarisation
            elif value == "Dilatation":
                self.dilatation()  # À remplir avec votre fonction de seuillage manuel
            elif value == "Ouverture":
                self.ouverture()  # À remplir avec votre fonction de seuillage OTSU
            elif value == "Fermeture":
                self.fermeture() 
            elif value == "Filtrage Morphologique":
                self.filtrage_morphologique()  # À remplir avec votre fonction de rotation
            elif value == "Croissance de regions":
                self.regionGrowing()  # À remplir avec votre fonction de redimension
            elif value == "Partition de regions":
                self.partitionD()  # À remplir avec votre fonction de dessin de rectangle
            elif value == "k means":
                self.KMeansSegmentation()  # À remplir avec votre fonction de binarisation
            elif value == "k means color":
                self.kmeans_color()  # À remplir avec votre fonction de binarisation
            elif value == "hough":
                self.Hough()  # À remplir avec votre fonction de seuillage manuel
            elif value == "sift":
                self.Sift()  # À remplir avec votre fonction de seuillage OTSU
            elif value == "Hariss":
                self.Hariss()  # À remplir avec votre fonction de seuillage OTSU
            elif value == "Huffman":
                self.compress_huffman()
            elif value == "LZW":
                self.compress_lzw()
            elif value == "Passe Bas":
                self.filter_haut_bas="bas"
                self.afficher_filtre()           
            elif value == "Passe Haut":
                self.filter_haut_bas="haut"
                self.afficher_filtre()           

import tkinter as tk
from tkinter import simpledialog

class FiltreDialog(simpledialog.Dialog):
    def __init__(self, parent, titre):
        self.titre = titre
        super().__init__(parent)

    def body(self, master):
        self.type_filtre = tk.StringVar()
        self.taille_filtre_rows = tk.StringVar()
        self.taille_filtre_cols = tk.StringVar()
        self.cutoff_frequency = tk.StringVar()
        self.butterworth_order = tk.StringVar()  # Variable pour stocker l'ordre du filtre Butterworth
        # Définir la valeur initiale de self.type_filtre sur "butterworth"
        self.type_filtre = tk.StringVar(value="gaussien")

        tk.Label(master, text="Sélection du filtre passe-{}".format(self.titre)).pack()

        cadre_type_filtre = tk.Frame(master)
        cadre_type_filtre.pack(pady=10)
        tk.Label(cadre_type_filtre, text="Choisir le type de filtre :").pack(side="left", padx=5)
        tk.Radiobutton(cadre_type_filtre, text="Butterworth", variable=self.type_filtre, value="butterworth", command=self.update_fields).pack(side="left", padx=5)
        tk.Radiobutton(cadre_type_filtre, text="Gaussien", variable=self.type_filtre, value="gaussien", command=self.update_fields).pack(side="left", padx=5)
        tk.Radiobutton(cadre_type_filtre, text="Idéal", variable=self.type_filtre, value="ideal", command=self.update_fields).pack(side="left", padx=5)

        cadre_cutoff_frequency = tk.Frame(master)
        cadre_cutoff_frequency.pack(pady=10)
        tk.Label(cadre_cutoff_frequency, text="Fréquence de coupure :").pack(side="left", padx=5)
        tk.Entry(cadre_cutoff_frequency, textvariable=self.cutoff_frequency, width=10).pack(side="left", padx=5)
        tk.Label(cadre_cutoff_frequency, text="Hz").pack(side="left")

        # Champ pour saisir l'ordre du filtre Butterworth, initialisé comme désactivé
        self.entry_order = tk.Entry(master, textvariable=self.butterworth_order, width=10, state="disabled")
        self.label_order = tk.Label(master, text="Ordre du filtre Butterworth:")
        self.label_order.pack(pady=5)
        self.entry_order.pack()

    def update_fields(self):
        # Fonction pour activer/désactiver le champ pour saisir l'ordre du filtre Butterworth
        if self.type_filtre.get() == "butterworth":
            self.entry_order.config(state="normal")
            self.label_order.config(fg="black")  # Change la couleur du texte pour indiquer que le champ est activé
        else:
            self.entry_order.delete(0, "end")
            self.entry_order.config(state="disabled")
            self.label_order.config(fg="gray")  # Change la couleur du texte pour indiquer que le champ est désactivé

    def apply(self):
        self.result = {
            "type_filtre": self.type_filtre.get(),
            "cutoff_frequency": self.cutoff_frequency.get(),
            "butterworth_order": self.butterworth_order.get() if self.type_filtre.get() == "butterworth" else None
        }


        
if __name__ == "__main__":
    app = App()
    app.mainloop()