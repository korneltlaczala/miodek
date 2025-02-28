import os
import customtkinter as ctk
from models import *

class App():
    def __init__(self):
        self.root = ctk.CTk()
        self.root.geometry("400x400")

        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.MODEL_DIR = os.path.join(self.BASE_DIR, "models")
        self.fill_frame()

        self.root.mainloop()

    def fill_frame(self):
        self.left_column = ctk.CTkFrame(self.root)
        self.left_column.pack(side='left', expand=True, fill='both', padx=5, pady=5)

        self.right_column = ctk.CTkFrame(self.root)
        self.right_column.pack(side='right', expand=True, fill='both', padx=5, pady=5)

        self.fill_left_column()

    def fill_left_column(self):
        self.data_files = self.get_data_files()
        self.model_files = self.get_models()

        self.data_file_label = ctk.CTkLabel(self.left_column, text="Select Data File:")
        self.data_file_label.pack(padx=5, pady=5)
        self.data_file_combobox = ctk.CTkComboBox(self.left_column, values=self.data_files, command=self.data_file_changed)
        self.data_file_combobox.pack(padx=5, pady=5)

        self.model_file_label = ctk.CTkLabel(self.left_column, text="Select Model File:")
        self.model_file_label.pack(padx=5, pady=5)
        self.model_file_combobox = ctk.CTkComboBox(self.left_column, values=self.model_files, command=self.model_file_changed)
        self.model_file_combobox.pack(padx=5, pady=5)

    def data_file_changed(self, value):
        print(f"selected data file: {value}")

    def model_file_changed(self, value):
        print(f"selected model file: {value}")



        # self.load_data_button = ctk.CTkButton(self.left_column, text="Load Data", command=self.load_data)
        # self.load_data_button.pack(padx=5, pady=5)


    def get_data_files(self):
        files = []
        for file in os.listdir(self.DATA_DIR):
            files.append(file)
        return files

    def get_models(self):
        models = []
        for file in os.listdir(self.MODEL_DIR):
            models.append(file)
        return models
    



if __name__ == "__main__":
    App()