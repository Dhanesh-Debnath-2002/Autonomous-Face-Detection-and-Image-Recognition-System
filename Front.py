from tkinter import *
from PIL import Image, ImageTk
from main import MainProject   # called different python class file to here

class Face_Recognition_system:
    def __init__(self,root):
        self.root = root
        self.root.geometry("1280x720+0+0")
        self.root.title("Autonomous Face Detection and Image Recognition Drone System")

        img = Image.open("Main.jpg")
        img = img.resize((1280,720), Image.LANCZOS)
        self.photoimg = ImageTk.PhotoImage(img)

        bg_img = Label(self.root,image=self.photoimg)
        bg_img.place(x=0, y=0, width=1280, height=720)

        img1 = Image.open("w.jpg")#change the photo
        img1 = img1.resize((200, 200), Image.LANCZOS)
        self.photoimg1 = ImageTk.PhotoImage(img1)

        b1 = Button(bg_img, image=self.photoimg1, command=self.mainproject, cursor="hand2")
        b1.place(x=565, y=170, width=150, height=140)

        b_1 = Button(bg_img, text="Click AFDIRDS Project", command=self.mainproject, cursor="hand2", font=("times new roman", 8, "bold"), bg="white", fg="black")
        b_1.place(x=565, y=307, width=150, height=30)

    def mainproject(self):
        self.new_window = Toplevel(self.root)
        self.app = MainProject(self.new_window)

if __name__ == "__main__":
    root = Tk()
    obj = Face_Recognition_system(root)
    root.mainloop()