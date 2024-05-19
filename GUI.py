from tkinter import *
from tkinter import messagebox 
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('placement_model.h5')

def predict():
  try:
    v_hsc_p = float(hsc_p.get())
    v_degree_p = float(degree_p.get())
    v_etest_p = float(etest_p.get())
    v_mba_p = float(mba_p.get())
    
    input_data = np.array([[v_hsc_p, v_degree_p, v_etest_p, v_mba_p]])
    
    prediction = model.predict(input_data)
    prediction = (prediction > 0.5).astype(int).flatten()[0]
    
    result = "Placed" if prediction == 1 else "Not Placed"
    messagebox.showinfo("Prediction Result", f"The predicted status is: {result}")
  except ValueError:
    messagebox.showerror("Input Error", "Please enter valid numeric values.")


font_main = 'Montserrat'
clr_main = '#FFA500'
clr_accent = '#FFC04C'

root = Tk()

frameForm = Frame(root, padx=10, pady=10, bg='white')
frameForm.pack(side=RIGHT, padx=10, pady=10,  fill=BOTH, expand=True)
lbInfo = Label(frameForm, text="Prediksi Penerimaan Kerja", font=(font_main, 14), fg=clr_main, bg=frameForm.cget('bg'))
lbInfo.pack(side=TOP, anchor='w')

frameField = Frame(frameForm, bg= frameForm.cget('bg'))

def validate_input(event, entry):
    current_text = entry.get()
    try:
        float(current_text)
    except ValueError:
        entry.delete(len(current_text) - 1, END)

hsc_p = Entry(frameField, bg=frameField.cget('bg'), font=(font_main, 12), width=30)
hsc_p.bind("<KeyRelease>", lambda event: validate_input(event, hsc_p))

degree_p = Entry(frameField, bg=frameField.cget('bg'), font=(font_main, 12), width=30)
degree_p.bind("<KeyRelease>", lambda event: validate_input(event, degree_p))

etest_p = Entry(frameField, bg=frameField.cget('bg'), font=(font_main, 12), width=30)
etest_p.bind("<KeyRelease>", lambda event: validate_input(event, etest_p))

mba_p = Entry(frameField, bg=frameField.cget('bg'), font=(font_main, 12), width=30)
mba_p.bind("<KeyRelease>", lambda event: validate_input(event, mba_p))

entryList = [
  ["Nilai SMA (HSC P)", hsc_p],
  ["Nilai Sarjana (Degree P)", degree_p],
  ["Nilai Test (ETest P)", etest_p],
  ["Nilai Magister (MBA P)", mba_p]
]

for i, entry in enumerate(entryList):
    lbField = Label(frameField, text=entry[0], bg=frameField.cget('bg'), font=(font_main, 12),anchor='w')
    lbField.grid(row=i, column=0, sticky='ew',padx=4, pady=8)
    Label(frameField, text=":", bg=frameField.cget('bg'), font=(font_main, 12)).grid(row=i, column=1)
    entry[1].grid(row=i, column=2, sticky='ew', padx=16, pady=4)

addBtn = Button(frameField, command=predict, text="Prediksi", padx=10, pady=2, borderwidth=0, bg=clr_main, fg='white',activebackground=clr_accent, activeforeground='white', font=(font_main, 12))
addBtn.grid(row=4, column=0, columnspan=3, sticky='ew', padx=16, pady=16) 
frameField.pack(side=TOP, padx=10, pady=10, fill='both', anchor='center')


root.mainloop()