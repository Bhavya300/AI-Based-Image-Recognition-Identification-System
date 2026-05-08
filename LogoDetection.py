import cv2
import numpy as np
import os
import imagehash  # Import imagehash for hashing the images
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import time

lpath = os.getcwd() + r"\\Logos\\"

logos = {
    "NIT Agartala(NITA)": [lpath + "agartala1.jpeg"],
    "NIT Allahabad(MNNIT)": [lpath + "allahabad1.jpeg", lpath + "allahabad2.jpeg", lpath + "allahabad3.png"],
    "NIT Andhra Pradesh ": [lpath + "andhra1.jpeg"],
    "NIT Arunachal Pradesh(NITAP)": [lpath + "arunachal1.jpeg"],
    "NIT Bhopal(MANIT)": [lpath + "bhopal1.png", lpath + "bhopal2.png"],
    "NIT Calicut(NITC)": [lpath + "calicut1.png"],
    "NIT Delhi(NITD)": [lpath + "delhi1.png"],
    "NIT Durgapur(NITDGP)": [lpath + "durgapur1.png", lpath + "durgapur5.png"],
    "NIT Goa(NITG)": [lpath + "goa1.png", lpath + "goa2.png"],
    "NIT Hamirpur(NITH)": [lpath + "hamirpur1.png"],
    "NIT Jaipur(MNIT)": [lpath + "jaipur1.jpeg"],
    "NIT Jalandhar(NITJ)": [lpath + "jalandhar1.png", lpath + "jalandhar2.png"],
    "NIT Jamshedpur(NITJSR)": [lpath + "jamshedpur1.png"],
    "NIT Kurukshetra(NITKKR)": [lpath + "Kurukshetra1.png", lpath + "Kurukshetra2.jpeg"],
    "NIT Manipur(NITMN)": [lpath + "manipur1.png"],
    "NIT Meghalaya(NITMGH)": [lpath + "meghalaya1.jpeg"],
    "NIT Mizoram(NITMZ)": [lpath + "mizoram1.jpeg"],
    "NIT Nagaland(NITN)": [lpath + "nagaland1.jpeg"],
    "NIT Nagpur(VNIT)": [lpath + "nagpur1.png", lpath + "nagpur2.jpeg", lpath + "nagpur3.jpeg"],
    "NIT Patna(NITP)": [lpath + "patna1.jpeg", lpath + "patna3.png",lpath + "patna5.jpeg"],
    "NIT Puducherry(NITPY)": [lpath + "puducherry1.jpeg"],
    "NIT Raipur(NITRR)": [lpath + "raipur1.jpeg",lpath + "raipur3.jpeg"],
    "NIT Rourkela(NITRKL)": [lpath + "rourkela1.png", lpath + "rourkela2.png",lpath+"rourkela3.png"],
    "NIT Sikkim(NITSKM)": [lpath + "sikkim1.png", lpath + "sikkim2.jpeg"],
    "NIT Silchar(NITS)": [lpath + "silchar1.png", lpath + "silchar2.png"],
    "NIT Srinagar": [lpath + "srinagar1.jpeg", lpath + "srinagar2.jpeg", lpath + "srinagar3.jpeg"],
    "NIT Surat(SVNIT)": [lpath + "surat1.png"],
    "NIT Surathkal(NITK)": [lpath + "surathkal1.png", lpath + "surathkal2.jpeg", lpath + "surathkal3.png"],
    "NIT Trichy(NITT)": [lpath + "trichy1.png"],
    "NIT Uttarakhand(NITUK)": [lpath + "uttarakhand1.jpeg"],
    "NIT Warangal(NITW)": [lpath + "warangal1.png",lpath+"warangal2.png",lpath + "warangal3.png", lpath + "warangal4.jpeg"]
}
def load_logo(logo_path):
    logo = cv2.imread(logo_path)
    return logo if logo is not None else None
def detect_logos_with_sift_flann(main_image):
    sift = cv2.SIFT_create() #detect and describe the keypoints in images
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    detected = 0

    min_good_matches = 15  # Increase for higher accuracy
    main_gray = Image.fromarray(cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY))

    for NIT_name, logo_paths in logos.items():
        for logo_path in logo_paths:  # Iterate over all logos for the NIT
            logo = load_logo(logo_path)
            if logo is None:
                continue

            logo_gray = Image.fromarray(cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY))
            hash0 = imagehash.average_hash(main_gray)
            hash1 = imagehash.average_hash(logo_gray)

            # Define a cutoff threshold for similarity
            cutoff = 1  # Adjust the cutoff based on the level of tolerance you need
            hashDiff = hash0 - hash1
            # Check similarity based on hash difference
            if hashDiff < cutoff:
                detected = 1
                return NIT_name

    if detected == 0:
        main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
        for NIT_name, logo_paths in logos.items():
            for logo_path in logo_paths:  # Iterate over all logos for the NIT
                logo = load_logo(logo_path)
                if logo is None:
                    continue

                logo_gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
                kp1, des1 = sift.detectAndCompute(logo_gray, None)
                kp2, des2 = sift.detectAndCompute(main_gray, None)

                if des1 is not None and des2 is not None:
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    matches = flann.knnMatch(des1, des2, k=2) # k nearest neighbour matching

                    good_matches = [m for m, n in matches if m.distance < 0.6 * n.distance]

                    if len(good_matches) > min_good_matches:
                        # Geometric verification using homography
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        matches_mask = mask.ravel().tolist()

                        if M is not None and sum(matches_mask) > min_good_matches:
                            return NIT_name

    return None

def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        main_image = cv2.imread(file_path)
        if main_image is not None:
            detected_NIT = detect_logos_with_sift_flann(main_image)

            # Display the loaded image
            image_rgb = cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            image_tk = ImageTk.PhotoImage(image_pil)
            label_image.config(image=image_tk)
            label_image.image = image_tk

            if detected_NIT:
                # Show the message box with the NIT name
                messagebox.showinfo("Detected the NIT", detected_NIT)
                label_image.config(image='')
            else:
                messagebox.showinfo("Result", "Can't detect the NIT.")
                label_image.config(image='')
        else:
            messagebox.showerror("Error", "Unable to load the image. Please check the path.")

def start_video_capture():
    cap = cv2.VideoCapture(0)  # Open the camera
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access the camera")
        return

    time_limit = 10 # Run the camera for 10 seconds
    start_time = time.time()  # Record the start time
    detected_NIT = None  # Variable to track if a logo is detected

    while True:
        ret, frame = cap.read()  #ret boolean to indicate if image is captured or not #frame is a numpy array
        if not ret:
            messagebox.showerror("Error", "Failed to capture image from the camera")
            break

        detected_NIT = detect_logos_with_sift_flann(frame)

        frame_resized = cv2.resize(frame, (400, 300))

        image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        label_image.config(image=image_tk)
        label_image.image = image_tk

        if detected_NIT:
            messagebox.showinfo("Detected NIT", detected_NIT)
            label_image.config(image='')  # Clear image label after detection
            break

        if time.time() - start_time > time_limit:
            break
    # Release the camera
    cap.release()
    label_image.config(image='')  # Clear the image label

    if not detected_NIT:
        messagebox.showinfo("error", "No NIT logo detected.")

def retrieve_logo():
    NIT_name = simpledialog.askstring("Input", "Enter the NIT name:")

    if NIT_name:
        # Search for NITs that contain the entered substring (case-insensitive)
        matching_NITs = [key for key in logos if NIT_name.lower() in key.lower()]

        if matching_NITs:
            if len(matching_NITs) == 1:
                selected_NIT = matching_NITs[0]
            else:
                # If more than one match, ask the user to select
                selected_NIT = simpledialog.askstring(
                    "Select the NIT",
                    f"Multiple similar NIT names found: {', '.join(matching_NITs)}. Please enter one(Enter the full name of the NIT):"
                )
                if selected_NIT not in matching_NITs:
                    messagebox.showerror("Error", "Invalid NIT name selected.")
                    return

            # Retrieve and show the logos for the selected NIT
            logo_paths = logos.get(selected_NIT)
            if logo_paths:
                for logo_path in logo_paths:
                    if os.path.exists(logo_path):
                        logo_image = cv2.imread(logo_path)

                        # Create a new window to display the logo
                        logo_window = tk.Toplevel(app)
                        logo_window.title(f"Logo for {selected_NIT}")

                        # Convert the logo image for Tkinter
                        logo_rgb = cv2.cvtColor(logo_image, cv2.COLOR_BGR2RGB)
                        logo_pil = Image.fromarray(logo_rgb) #converts numpy array to PIL image
                        logo_tk = ImageTk.PhotoImage(logo_pil)

                        # Create a label in the new window to display the logo
                        label_logo = tk.Label(logo_window, image=logo_tk)
                        label_logo.pack(padx=10, pady=10)

                        label_logo.image = logo_tk  # Keep a reference to avoid garbage collection
                        break  # Only show the first logo
            else:
                messagebox.showerror("Error", "No logos found for this NIT.")
        else:
            messagebox.showerror("Error", "wrong NIT name given.")


app = tk.Tk()
app.title("Logo Detection: Static & Real-Time")
app.geometry("800x600")

background_image = Image.open(os.getcwd()+r"\\back_pic.jpg")
background_image = background_image.resize((1300, 650))
background_image_tk = ImageTk.PhotoImage(background_image)
app.background_image = background_image_tk  # Keep a reference
background_label = tk.Label(app, image=app.background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

title_label = tk.Label(app, text="NIT Logo Finder", font=("Castellar", 24), bg="white", fg="black")
title_label.pack(pady=30)

label_image = tk.Label(app,bg='light grey')
label_image.pack(pady=20)

button_load = tk.Button(app, text="Load Image of NIT for Logo Detection", command=load_image)
button_load.pack(pady=20)

button_start = tk.Button(app, text="Start Real-Time NIT Logo Detection", command=start_video_capture)
button_start.pack(pady=20)

button_retrieve = tk.Button(app, text="Retrieve Logo by entering the NIT name", command=retrieve_logo)
button_retrieve.pack(pady=20)

button_exit = tk.Button(app, text="Exit from the program", command=app.quit)
button_exit.pack(pady=20)

app.mainloop()