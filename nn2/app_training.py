import os
from tkinter import messagebox
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from models import *
from training import *

class App:
    def __init__(self, trainer_name):
        self.root = ctk.CTk()
        self.root.geometry("1400x800")

        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.MODEL_DIR = os.path.join(self.BASE_DIR, "models")

        self.env = TrainingEnvironment(self, trainer_name, epochs=1e4, learning_rate=0.1, report_interval=1000)
        # self.env = TrainingEnvironment.load()
        self.trainer_frame = TrainerFrame(self.root, self)
        self.trainer_frame.pack(side='top', expand=False, fill='both', padx=5, pady=5)
        self.result_frame = ResultFrame(self.root, self)
        self.result_frame.pack(side='bottom', expand=True, fill='both', padx=5, pady=5)

        self.root.mainloop()

class TrainerFrame(ctk.CTkFrame):

    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.app = app
        self.fill()

    def fill(self):
        self.command_frame = ctk.CTkFrame(self)
        self.progress_frame = ctk.CTkFrame(self)
        self.settings_frame = ctk.CTkFrame(self)
        self.command_frame.grid(row=0, column=0, rowspan=2, padx=5, pady=5, sticky='nsew')
        self.progress_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky='nsew')
        self.settings_frame.grid(row=0, column=2, padx=5, pady=5, sticky='nsew')
        
        self.fill_command_frame()
        self.fill_progress_frame()
        self.fill_settings_frame()

    def fill_command_frame(self):
        self.trainer_name_entry = ctk.CTkEntry(self.command_frame, placeholder_text="Trainer Name", font=("Helvetica", 16))
        self.trainer_name_entry.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')

        self.undo_training_button = ctk.CTkButton(self.command_frame, text="Undo Training", command=self.env.undo_training)
        self.train_button = ctk.CTkButton(self.command_frame, text="Train", command=self.env.train)
        self.undo_training_button.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
        self.train_button.grid(row=1, column=1, padx=5, pady=5)

        self.save_best_button = ctk.CTkButton(self.command_frame, text="Save Best", command=self.env.save_best)
        self.save_last_button = ctk.CTkButton(self.command_frame, text="Save Last", command=self.env.save_last)
        self.save_best_button.grid(row=2, column=0, padx=5, pady=5, sticky='nsew')
        self.save_last_button.grid(row=2, column=1, padx=5, pady=5, sticky='nsew')

        self.update_command_frame()

    def fill_progress_frame(self):
        self.current_epoch_label = ctk.CTkLabel(self.progress_frame, anchor='w')
        self.current_epoch_label.grid(row=0, column=2, padx=15, pady=2, sticky='nsew')
        self.current_mse_label = ctk.CTkLabel(self.progress_frame, anchor='w')
        self.current_mse_label.grid(row=1, column=2, padx=15, pady=2, sticky='nsew')
        self.best_mse_label = ctk.CTkLabel(self.progress_frame, anchor='w')
        self.best_mse_label.grid(row=2, column=2, padx=15, pady=2, sticky='nsew')
        self.update_progress_frame()

    def fill_settings_frame(self):
        self.learning_rate_label = ctk.CTkLabel(self.settings_frame, anchor='w')
        self.learning_rate_label.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        self.generate_learning_rate_buttons()

        self.new_epochs_label = ctk.CTkLabel(self.settings_frame, anchor='w')
        self.new_epochs_label.grid(row=2, column=0, padx=5, pady=5, sticky='nsew')
        self.generate_epoch_settings()

        self.update_settings_frame()

    def update_command_frame(self):
        self.trainer_name_entry.delete(0, 'end')
        self.trainer_name_entry.insert(0, self.env.trainer_name)

    def update_progress_frame(self):
        current_epoch = str(self.env.trainer.current_epoch)
        mse = str(round(self.env.trainer.tester.mse, 2))
        best_mse = str(round(self.env.trainer.best_mse, 2))
        self.current_epoch_label.configure(text="Current Epoch: " + current_epoch)
        self.current_mse_label.configure(text="Current MSE:\t" + mse)
        self.best_mse_label.configure(text="Best MSE:\t" + best_mse)

    def update_settings_frame(self):
        self.learning_rate_label.configure(text=f"Learning Rate: {self.env.learning_rate}")
        self.new_epochs_label.configure(text=f"New Epochs: {self.env.epochs}")

    def generate_learning_rate_buttons(self):
        learning_rates = [1/10**i for i in range(1, 7)]
        self.learning_rate_buttons = []
        for i, rate in enumerate(learning_rates):
            button = ctk.CTkButton(self.settings_frame, text=str(rate), 
                                   command=lambda r=rate: self.env.set_learning_rate(r))
            button.grid(row=1, column=i, padx=5, pady=5, sticky='nsew')
            self.learning_rate_buttons.append(button)

    def generate_epoch_settings(self):
        values = [str(i) for i in range(1, 11)]
        self.epoch_multiplier_combobox = ctk.CTkComboBox(self.settings_frame, values=values, command=self.env.set_epoch_multiplier)
        self.epoch_multiplier_combobox.grid(row=3, column=0, padx=5, pady=5, sticky='nsew')

        epoch_choices = [10**i for i in range(3, 7)]
        self.epoch_choices_buttons = []
        for i, choice in enumerate(epoch_choices):
            button = ctk.CTkButton(self.settings_frame, text=str(choice), 
                                   command=lambda c=choice: self.env.set_epoch_oom(c))
            button.grid(row=3, column=i+1, padx=5, pady=5, sticky='nsew')
            self.epoch_choices_buttons.append(button)

    @property
    def env (self):
        return self.app.env

class ResultFrame(ctk.CTkFrame):

    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, **kwargs)
        self.app = app
        self.fill()

    def fill(self):
        self.fit_frame = ctk.CTkFrame(self)
        self.fit_frame.pack(padx=5, pady=5, fill='both')
        self.history_frame = ctk.CTkFrame(self)
        self.history_frame.pack(padx=5, pady=5, fill='both')

        self.fit_frame.grid_columnconfigure(0, weight=1)
        self.fit_frame.grid_columnconfigure(1, weight=1)
        self.fit_frame.grid_columnconfigure(2, weight=1)

        self.init_fit_canvas = FigureCanvasTkAgg(master=self.fit_frame)
        self.best_fit_canvas = FigureCanvasTkAgg(master=self.fit_frame)
        self.last_fit_canvas = FigureCanvasTkAgg(master=self.fit_frame)
        self.init_fit_canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        self.best_fit_canvas.get_tk_widget().grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
        self.last_fit_canvas.get_tk_widget().grid(row=0, column=2, padx=5, pady=5, sticky='nsew')

        # self.init_fit_info_frame = ctk.CTkFrame(self.fit_frame)
        # self.best_fit_info_frame = ctk.CTkFrame(self.fit_frame)
        # self.last_fit_info_frame = ctk.CTkFrame(self.fit_frame)

        self.init_fit_label = ctk.CTkLabel(self.fit_frame, text="Model before training", font=("Helvetica", 18, "bold"), anchor='w')
        self.best_fit_label = ctk.CTkLabel(self.fit_frame, text="Best model so far", font=("Helvetica", 18, "bold"), anchor='w')
        self.last_fit_label = ctk.CTkLabel(self.fit_frame, text="Model after training", font=("Helvetica", 18, "bold"), anchor='w')
        self.init_fit_label.grid(row=1, column=0, padx=20, pady=10, sticky='nsew')
        self.best_fit_label.grid(row=1, column=1, padx=20, pady=10, sticky='nsew')
        self.last_fit_label.grid(row=1, column=2, padx=20, pady=10, sticky='nsew')

        self.init_fit_epoch_label = ctk.CTkLabel(self.fit_frame, anchor='w')
        self.best_fit_epoch_label = ctk.CTkLabel(self.fit_frame, anchor='w')
        self.last_fit_epoch_label = ctk.CTkLabel(self.fit_frame, anchor='w')
        self.init_fit_epoch_label.grid(row=3, column=0, padx=20, pady=2, sticky='nsew')
        self.best_fit_epoch_label.grid(row=3, column=1, padx=20, pady=2, sticky='nsew')
        self.last_fit_epoch_label.grid(row=3, column=2, padx=20, pady=2, sticky='nsew')

        self.init_fit_mse_label = ctk.CTkLabel(self.fit_frame, anchor='w')
        self.best_fit_mse_label = ctk.CTkLabel(self.fit_frame, anchor='w')
        self.last_fit_mse_label = ctk.CTkLabel(self.fit_frame, anchor='w')
        self.init_fit_mse_label.grid(row=4, column=0, padx=20, pady=2, sticky='nsew')
        self.best_fit_mse_label.grid(row=4, column=1, padx=20, pady=2, sticky='nsew')
        self.last_fit_mse_label.grid(row=4, column=2, padx=20, pady=2, sticky='nsew')

        self.update()

    def update_canvas(self, canvas, model):
        canvas.figure.clear()
        canvas.figure = self.env.trainer.get_fig(model)
        canvas.draw()

    def update(self):
        self.update_canvas(self.init_fit_canvas, self.env.trainer.init_model)
        self.update_canvas(self.best_fit_canvas, self.env.trainer.best_model)
        self.update_canvas(self.last_fit_canvas, self.env.trainer.model)

        self.init_fit_epoch_label.configure(text=f"Epoch:\t {self.env.trainer.init_model_age}")
        self.best_fit_epoch_label.configure(text=f"Epoch:\t {self.env.trainer.best_model_age}")
        self.last_fit_epoch_label.configure(text=f"Epoch:\t{self.env.trainer.current_epoch}")
        self.init_fit_mse_label.configure(text=f"MSE:\t{round(self.env.trainer.init_mse, 2)}")
        self.best_fit_mse_label.configure(text=f"MSE:\t{round(self.env.trainer.best_mse, 2)}")
        self.last_fit_mse_label.configure(text=f"MSE:\t{round(self.env.trainer.mse, 2)}")

    @property
    def env(self):
        return self.app.env

class TrainingEnvironment():

    def __init__(self, app, trainer_name, epochs, learning_rate, report_interval):
        self.app = app
        self.learning_rate = learning_rate
        self.report_interval = report_interval
        self.set_trainer(trainer_name)
        self.setup_epochs(epochs)

    def set_trainer(self, trainer_name=None):
        if trainer_name is not None:
            self.trainer_name = trainer_name
        self.trainer = Trainer.load(self.trainer_name)

    def setup_epochs(self, epochs):
        self.epochs = epochs
        self.epoch_multiplier = 1
        self.epoch_oom = epochs
        self.app

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        self.app.trainer_frame.update_settings_frame()

    def set_epoch_multiplier(self, value):
        self.epoch_multiplier = int(value)
        self.set_epochs()

    def set_epoch_oom(self, epoch_oom):
        self.epoch_oom = epoch_oom
        self.set_epochs()

    def set_epochs(self):
        self.epochs = self.epoch_multiplier * self.epoch_oom
        self.app.trainer_frame.update_settings_frame()

    def train(self):
        self.trainer.train_in_app(epochs=self.epochs,
                           learning_rate=self.learning_rate,
                           report_interval=self.report_interval)
        self.app.trainer_frame.update_progress_frame()
        self.app.results_frame.update_result_frame()

    def save_last(self):
        self.trainer.save(self.get_new_trainer_name())
        self.update_app()

    def save_best(self):
        self.trainer.save_with_best_model(self.get_new_trainer_name())
        self.update_app()

    def undo_training(self):
        self.trainer.undo_training()
        self.update_app()

    def update_app(self):
        self.app.trainer_frame.update_progress_frame()
        self.app.results_frame.update()

    def get_new_trainer_name(self):
        return self.app.trainer_frame.trainer_name_entry.get()

    def save(self):
        trainer = self.trainer
        self.trainer = None
        with open(f'app/training_env.pkl', 'wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
        self.trainer = trainer

    @classmethod
    def load(self):
        with open(f'app/training_env.pkl', 'rb') as file:
            return pickle.load(file)
        
if __name__ == "__main__":
    App(trainer_name="trainer_square")
    # App(trainer_name="trainer_app")
#   # App()