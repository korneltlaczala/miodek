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

        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(expand=True, fill='both')

        self.edit_model_frame = ctk.CTkFrame(self.root)
        self.edit_model_frame.pack_forget()

        self.fill_main_frame()
        self.root.mainloop()

    def fill_main_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        self.left_column = ctk.CTkFrame(self.main_frame)
        self.left_column.pack(side='left', expand=True, fill='both', padx=5, pady=5)

        self.right_column = ctk.CTkFrame(self.main_frame)
        self.right_column.pack(side='right', expand=True, fill='both', padx=5, pady=5)

        self.fill_left_column()
        self.set_defaults()

    def fill_left_column(self):
        self.data_files = self.get_data_files()
        self.model_files = self.get_model_files()

        self.data_file_label = ctk.CTkLabel(self.left_column, text="Select Dataset:", font=("Helvetica", 16, "bold"), anchor='w')
        self.data_file_label.pack(padx=5, pady=5, fill='x')
        self.data_file_combobox = ctk.CTkComboBox(self.left_column, values=self.data_files, command=self.data_file_changed)
        self.data_file_combobox.pack(padx=5, pady=5, fill='x')

        self.model_file_label = ctk.CTkLabel(self.left_column, text="Select Model:", font=("Helvetica", 16, "bold"), anchor='w')
        self.model_file_label.pack(padx=5, pady=5, fill='x')
        self.model_file_combobox = ctk.CTkComboBox(self.left_column, values=self.model_files, command=self.model_file_changed)
        self.model_file_combobox.pack(padx=5, pady=5, fill='x')

        self.new_model_button = ctk.CTkButton(self.left_column, text="New Model", command=self.new_model)
        self.new_model_button.pack(padx=5, pady=5, fill='x')

        self.mse_label = ctk.CTkLabel(self.left_column, text="MSE:", font=("Helvetica", 16, "bold"), anchor='w')
        self.mse_label.pack(padx=5, pady=(0,5), fill='x')

        self.edit_model_button = ctk.CTkButton(self.left_column, text="Edit Model", command=self.edit_model)
        self.edit_model_button.pack(padx=5, pady=5, fill='x')

    def set_defaults(self):
        data_files = self.get_data_files()
        model_files = self.get_model_files()
        if len(model_files) == 0:
            model = MLP(1, [5], 1)
            self.save_model(model=model, name="new_model")
            model_files = self.get_model_files()
            
        self.data_file_changed(data_files[0])
        self.model_file_changed(model_files[0])

    def new_model(self):
        model = MLP(1, [5], 1)
        self.edit_model_view(model)

    def edit_model(self):
        self.edit_model_view(self.tester.model.get_copy())

    def edit_model_view(self, model):
        self.fill_edit_model_frame(model)

        self.main_frame.pack_forget()
        self.edit_model_frame.pack(expand=True, fill='both')

    def fill_edit_model_frame(self, model):
        for widget in self.edit_model_frame.winfo_children():
            widget.destroy()

        self.model_name_label = ctk.CTkLabel(self.edit_model_frame, text="Model Name:", font=("Helvetica", 16, "bold"), anchor='w')
        self.model_name_label.pack(padx=5, pady=5, fill='x')
        self.model_name_entry = ctk.CTkEntry(self.edit_model_frame, placeholder_text="Model Name")
        self.model_name_entry.pack(padx=5, pady=5, fill='x')

        self.save_model_button = ctk.CTkButton(self.edit_model_frame, text="Save Model", command=lambda: self.save_model(model, self.model_name_entry.get()))
        self.save_model_button.pack(pady=20)
        self.cancel_button = ctk.CTkButton(self.edit_model_frame, text="Cancel", command=self.return_to_main_view)
        self.cancel_button.pack(pady=20)

        self.model_details_frame = ctk.CTkFrame(self.edit_model_frame)
        self.model_details_frame.pack(padx=5, pady=5, fill='both')
        self.neuron_weights_frame = ctk.CTkFrame(self.edit_model_frame)
        self.neuron_weights_frame.pack(padx=5, pady=5, fill='both')

        self.create_neuron_buttons(model)

    def create_neuron_buttons(self, model):
        for widget in self.model_details_frame.winfo_children():
            widget.destroy()

        def create_layer_frame(parent):
            frame = ctk.CTkFrame(parent)
            frame.pack(side='left', expand=True, fill='both', padx=5, pady=5)
            # Vertical centering support
            frame.grid_rowconfigure(0, weight=1)  # Top spacer (pushes everything down)
            frame.grid_columnconfigure(0, weight=1)  # Left spacer (centers horizontally)
            frame.grid_columnconfigure(2, weight=1)  # Right spacer (centers horizontally)
            return frame

        # Input layer (first column)
        input_layer_frame = create_layer_frame(self.model_details_frame)

        for neuron_index in range(model.inputs):
            neuron_button = ctk.CTkButton(input_layer_frame, text=str(neuron_index), 
                                        command=lambda n=neuron_index: self.neuron_button_clicked(0, n))
            neuron_button.configure(corner_radius=50, width=50, height=50)
            neuron_button.grid(row=neuron_index+1, column=1, pady=5)  # Centered in column 1

        # Add top and bottom spacers to vertically center buttons
        input_layer_frame.grid_rowconfigure(model.inputs + 1, weight=1)  # Bottom spacer

        # Hidden layers
        for layer_index, layer in enumerate(model.layers):
            layer_frame = create_layer_frame(self.model_details_frame)

            for neuron_index in range(layer.neurons_out):
                neuron_button = ctk.CTkButton(layer_frame, text=str(neuron_index), 
                                            command=lambda l=layer_index, n=neuron_index: self.neuron_button_clicked(l, n))
                neuron_button.configure(corner_radius=50, width=50, height=50)
                neuron_button.grid(row=neuron_index+1, column=1, pady=5)  # Centered horizontally (column 1)

            # Add top and bottom spacers to vertically center buttons
            layer_frame.grid_rowconfigure(layer.neurons_out + 1, weight=1)  # Bottom spacer

    def neuron_button_clicked(self, layer_index, neuron_index):
        layer = self.tester.model.layers[layer_index]
        self.fill_neuron_weights_frame(self.tester.model, layer_index, neuron_index)

    def fill_neuron_weights_frame(self, model, layer_index, neuron_index):
        for widget in self.neuron_weights_frame.winfo_children():
            widget.destroy()

        layer = model.layers[layer_index]
        weight_frame = ctk.CTkFrame(self.neuron_weights_frame)
        weight_frame.pack(padx=5, pady=5, fill='x')
        for weight_index in range(layer.neurons_in):
            # print(layer.weights[neuron_index, weight_index])
            weight_entry = ctk.CTkEntry(weight_frame, placeholder_text=str(layer.weights[neuron_index, weight_index]), width=50)
            weight_entry.pack(side='left', padx=5, pady=5)

    def return_to_main_view(self):
        self.edit_model_frame.pack_forget()
        self.fill_main_frame()
        self.main_frame.pack(expand=True, fill='both')

    def save_model(self, model, name):
        model_path = os.path.join(self.MODEL_DIR, name)
        if os.path.exists(model_path):
            if self.ask_overwrite() == False:
                return
        model.save(name)
        self.return_to_main_view()

    def ask_overwrite(self):
        response = input("File already exists, overwrite? (y/n) ")
        while response not in ['y', 'n']:
            response = input("Invalid input, please try again: ")
        return response == 'y'

    def data_file_changed(self, value):
        data_file = os.path.join(self.DATA_DIR, value)
        self.tester.set_data_file(data_file)
        self.run_model()

    def model_file_changed(self, value):
        model_path = os.path.join(self.MODEL_DIR, value)
        model = MLP.load(model_path)
        print(model)
        self.tester.set_model(model)
        self.run_model()

    def run_model(self):
        if(self.tester.run()):
            mse_rounded = round(self.tester.mse, 2)
            self.mse_label.configure(text=f"MSE: {mse_rounded}")
            self.plot()

    def plot(self):
        if self.plot_canvas is not None:
            self.plot_canvas.get_tk_widget().destroy()
        self.plot_canvas = FigureCanvasTkAgg(self.tester.get_fig(), master=self.right_column)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(expand=True, fill='both')


    def get_data_files(self):
        files = []
        for file in os.listdir(self.DATA_DIR):
            files.append(file)
        return files

    def get_model_files(self):
        models = []
        for file in os.listdir(self.MODEL_DIR):
            models.append(file)
        return models
    

if __name__ == "__main__":
    App()