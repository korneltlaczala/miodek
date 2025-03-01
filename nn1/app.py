import os
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from models import *

class App():
    def __init__(self):
        self.root = ctk.CTk()
        self.root.geometry("1000x600")

        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.MODEL_DIR = os.path.join(self.BASE_DIR, "models")
        self.plot_canvas = None
        self.tester = Tester()
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

        self.data_file_label = ctk.CTkLabel(self.left_column, text="Select Data File:", font=("Helvetica", 16, "bold"), justify='left')
        self.data_file_label.pack(padx=5, pady=5, fill='x')
        self.data_file_combobox = ctk.CTkComboBox(self.left_column, values=self.data_files, command=self.data_file_changed)
        self.data_file_combobox.pack(padx=5, pady=5, fill='x')

        self.model_file_label = ctk.CTkLabel(self.left_column, text="Select Model File:", font=("Helvetica", 16, "bold"), justify='left')
        self.model_file_label.pack(padx=5, pady=5, fill='x')
        self.model_file_combobox = ctk.CTkComboBox(self.left_column, values=self.model_files, command=self.model_file_changed)
        self.model_file_combobox.pack(padx=5, pady=5, fill='x')

    def data_file_changed(self, value):
        data_file = os.path.join(self.DATA_DIR, value)
        self.tester.set_data_file(data_file)

        if(self.tester.run()):
            self.tester.report()
            self.plot()
        print(self.tester.ready)

    def model_file_changed(self, value):
        model_path = os.path.join(self.MODEL_DIR, value)
        model = MLP.load(model_path)
        self.tester.set_model(model)

        if(self.tester.run()):
            self.tester.report()
            self.plot()
        print(self.tester.ready)

    def plot(self):
        if self.plot_canvas is not None:
            self.plot_canvas.get_tk_widget().destroy()
        self.plot_canvas = FigureCanvasTkAgg(self.tester.get_fig(), master=self.right_column)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(expand=True, fill='both')


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