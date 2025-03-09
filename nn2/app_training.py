import os
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

        self.env = TrainingEnvironment(self, trainer_name, epochs=5e4, learning_rate=0.1, report_interval=1000)
        self.trainer_frame = TrainerFrame(self.root, self, self.env)
        self.trainer_frame.pack(side='top', expand=False, fill='both', padx=5, pady=5)
        self.result_frame = ResultFrame(self.root, self)
        self.result_frame.pack(side='bottom', expand=True, fill='both', padx=5, pady=5)

        self.root.mainloop()


class TrainerFrame(ctk.CTkFrame):

    def __init__(self, parent, app, env, **kwargs):
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.app = app
        self.env = env
        self.fill()


    def fill(self):
        self.save_frame = ctk.CTkFrame(self)
        self.progress_frame = ctk.CTkFrame(self)
        self.settings_frame = ctk.CTkFrame(self)
        self.button_frame = ctk.CTkFrame(self)
        self.save_frame.grid(row=0, column=0, rowspan=2, padx=5, pady=5, sticky='nsew')
        self.progress_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky='nsew')
        self.settings_frame.grid(row=0, column=2, padx=5, pady=5, sticky='nsew')
        self.button_frame.grid(row=1, column=2, padx=5, pady=5, sticky='nsew')
        
        self.fill_save_frame()
        self.fill_progress_frame()
        self.fill_settings_frame()
        self.fill_button_frame()

    def fill_save_frame(self):
        self.trainer_name_entry = ctk.CTkEntry(self.save_frame, placeholder_text="Trainer Name", font=("Helvetica", 16))
        self.trainer_name_entry.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')
        self.save_button = ctk.CTkButton(self.save_frame, text="Save Trainer", command=self.env.save_trainer)
        self.save_button.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
        self.restore_button = ctk.CTkButton(self.save_frame, text="Restore Trainer", command=self.env.restore_trainer)
        self.restore_button.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')
        self.update_save_frame()

    def fill_progress_frame(self):
        current_epoch = str(self.trainer.current_epoch)
        mse = str(round(self.trainer.tester.mse, 2))
        best_mse = str(self.trainer.best_mse)
        self.current_epoch_label = ctk.CTkLabel(self.progress_frame, text="Current Epoch: " + current_epoch, anchor='w')
        self.current_epoch_label.grid(row=0, column=2, padx=5, pady=5, sticky='nsew')
        self.current_mse_label = ctk.CTkLabel(self.progress_frame, text="Current MSE: " + mse, anchor='w')
        self.current_mse_label.grid(row=1, column=2, padx=5, pady=5, sticky='nsew')
        self.best_mse_label = ctk.CTkLabel(self.progress_frame, text="Best MSE: " + best_mse, anchor='w')
        self.best_mse_label.grid(row=2, column=2, padx=5, pady=5, sticky='nsew')

    def fill_settings_frame(self):
        self.learning_rate_label = ctk.CTkLabel(self.settings_frame, anchor='w')
        self.learning_rate_label.configure(text=f"Learning Rate: {self.trainer.learning_rate}")
        self.learning_rate_label.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        self.generate_learning_rate_buttons()

        self.new_epochs_label = ctk.CTkLabel(self.settings_frame, anchor='w')
        self.new_epochs_label.configure(text=f"New Epochs: {self.trainer.new_epochs}")
        self.new_epochs_label.grid(row=2, column=0, padx=5, pady=5, sticky='nsew')
        self.generate_epoch_settings()

    def fill_button_frame(self):
        self.start_button = ctk.CTkButton(self.button_frame, text="Start", command=self.env.start_training)
        self.start_button.pack(side='left', padx=5, pady=5, fill='x')
 
    def update_save_frame(self):
        self.trainer_name_entry.delete(0, 'end')
        self.trainer_name_entry.insert(0, self.trainer.name)

    def update_progress_frame(self):
        current_epoch = str(self.trainer.current_epoch)
        mse = str(round(self.trainer.tester.mse, 2))
        # self.current_epoch_label.configure(text="Current Epoch: " + current_epoch)
        # self.current_mse_label.configure(text="Current MSE: " + mse)
        print("updating")

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
        self.epoch_multiplier_combobox = ctk.CTkComboBox(self.settings_frame, values=values, command=self.env.set_epoch_multiplier())
        self.epoch_multiplier_combobox.grid(row=3, column=0, padx=5, pady=5, sticky='nsew')

        epoch_choices = [10**i for i in range(3, 7)]
        self.epoch_choices_buttons = []
        for i, choice in enumerate(epoch_choices):
            button = ctk.CTkButton(self.settings_frame, text=str(choice), 
                                   command=lambda c=choice: self.env.set_epoch_oom(c))
            button.grid(row=3, column=i+1, padx=5, pady=5, sticky='nsew')
            self.epoch_choices_buttons.append(button)

        self.epoch_multiplier = int(self.epoch_multiplier_combobox.get())
        self.epoch_oom = epoch_choices[0]

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

        self.old_fit_canvas = FigureCanvasTkAgg(self.get_fit_fig(), master=self.fit_frame)
        self.new_fit_canvas = FigureCanvasTkAgg(self.get_fit_fig(), master=self.fit_frame)
        self.update_old_fit()
        self.update_new_fit()

    def update_old_fit(self):
        self.old_fit_canvas.get_tk_widget().destroy()
        self.old_fit_canvas = FigureCanvasTkAgg(self.get_fit_fig(), master=self.fit_frame)
        self.old_fit_canvas.draw()
        self.old_fit_canvas.get_tk_widget().pack(side='left', fill='both', expand=True)

    def update_new_fit(self):
        self.new_fit_canvas.get_tk_widget().destroy()
        self.new_fit_canvas = FigureCanvasTkAgg(self.get_fit_fig(), master=self.fit_frame)
        self.new_fit_canvas.draw()
        self.new_fit_canvas.get_tk_widget().pack(side='right', fill='both', expand=True)
    
    def get_fit_fig(self):
        return self.app.trainer.tester.get_fig(linear=True)


class TrainingEnvironment():

    def __init__(self, app, trainer_name, epochs, learning_rate, report_interval):
        self.app = app
        self.learning_rate = learning_rate
        self.report_interval = report_interval
        self.set_trainer(trainer_name)
        self.setup_epochs(epochs)

    def set_trainer(self, trainer_name):
        self.trainer_name = trainer_name
        self.trainer = Trainer.load(self.trainer_name)

    def setup_epochs(self, epochs):
        self.epochs = epochs
        self.epoch_multiplier = 1
        self.epoch_oom = epochs

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        self.app.trainer_frame.learning_rate_label.configure(text=f"Learning Rate: {self.learning_rate}")

    def set_epoch_multiplier(self):
        self.epoch_multiplier = int(self.app.trainer_frame.epoch_multiplier_combobox.get())
        self.set_new_epochs()

    def set_epoch_oom(self, epoch_oom):
        self.epoch_oom = epoch_oom
        self.set_new_epochs()

    def set_new_epochs(self):
        self.new_epochs = self.epoch_multiplier * self.epoch_oom
        self.app.trainer_frame.new_epochs_label.configure(text=f"New Epochs: {self.new_epochs}")

    def start_training(self):
        self.app.trainer.train(epochs=self.new_epochs,
                               learning_rate=self.learning_rate,
                               report_interval=self.report_interval)

    def save_trainer(self):
        new_name = self.app.trainer_frame.trainer_name_entry.get()
        self.trainer.save(new_name)

    def restore_trainer(self):
        self.trainer = Trainer.load(self.trainer_name)
        self.trainer_frame.destroy()
        self.trainer_frame = TrainerFrame(self.root, self, self.app.trainer)
        self.trainer_frame.pack(side='top', expand=False, fill='both', padx=5, pady=5)

if __name__ == "__main__":
    App(trainer_name="trainer_app")