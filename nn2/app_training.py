import os
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from models import *
from training import *

class App:
    def __init__(self, trainer_name):
        self.root = ctk.CTk()
        self.root.geometry("1000x600")

        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.MODEL_DIR = os.path.join(self.BASE_DIR, "models")

        # self.main_frame = ctk.CTkFrame(self.root)
        # self.main_frame.pack(expand=True, fill='both')

        self.trainer_name = trainer_name
        trainer = Trainer.load(trainer_name)
        self.trainer_frame = TrainerFrame(self.root, trainer)
        self.trainer_frame.pack(side='top', expand=False, fill='both', padx=5, pady=5)
        self.result_frame = ResultFrame(self.root)
        self.result_frame.pack(side='bottom', expand=True, fill='both', padx=5, pady=5)

        self.root.mainloop()

class TrainerFrame(ctk.CTkFrame):

    def __init__(self, parent, trainer, **kwargs):
        super().__init__(parent, **kwargs)
        self.trainer = trainer
        self.fill()


    def fill(self):

        self.grid_rowconfigure(0, weight=1)
        # self.grid_columnconfigure(0, weight=1)
        # self.grid_columnconfigure(1, weight=1)
        # self.grid_columnconfigure(2, weight=1)
        self.trainer_name_entry = ctk.CTkEntry(self, placeholder_text=self.trainer.name, font=("Helvetica", 16))
        self.trainer_name_entry.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')

        self.save_button = ctk.CTkButton(self, text="Save Trainer", command=lambda: self.trainer.save())
        self.save_button.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')

        self.restore_button = ctk.CTkButton(self, text="Restore Trainer", command=self.restore_trainer)
        self.restore_button.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')

        current_epoch = str(self.trainer.current_epoch)
        mse = str(round(self.trainer.tester.mse, 2))
        self.current_epoch_label = ctk.CTkLabel(self, text="Current Epoch: " + current_epoch, anchor='w')
        self.current_epoch_label.grid(row=0, column=2, padx=5, pady=5, sticky='nsew')
        self.current_mse_label = ctk.CTkLabel(self, text="Current MSE: " + mse, anchor='w')
        self.current_mse_label.grid(row=1, column=2, padx=5, pady=5, sticky='nsew')
        
    def restore_trainer(self):
        pass

class ResultFrame(ctk.CTkFrame):

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.title = ctk.CTkLabel(self, text="Results", font=("Helvetica", 16, "bold"), anchor='w')
        self.title.pack(padx=5, pady=5, fill='x')
        self.text = ctk.CTkLabel(self, text="Results will go here", anchor='w')
        self.text.pack(padx=5, pady=5, fill='x')


if __name__ == "__main__":
    App(trainer_name="trainer_square")