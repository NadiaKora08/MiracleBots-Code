import tkinter as tk
from tkinter import font
import pandas
import time
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import asyncio
from bleak import BleakClient, BleakScanner
import threading
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft
from PIL import Image, ImageTk
import sys
import os

# def resource_path(relative_path):
#     """ Get absolute path to resource, works for dev and for PyInstaller """
#     try:
#         base_path = sys._MEIPASS  # This is used by PyInstaller
#     except AttributeError:
#         base_path = os.path.abspath(".")  # This is used during development
#     return os.path.join(base_path, relative_path)


def process_bp_signal(signal_input, time_input):
    time1 = np.array(time_input)
    signal1 = np.array(signal_input)
    time_f = time1 - time1[0]
    print(time_f)
    print(signal1)

    t_M = max(time_f)  # max time
    t_L = len(time_f)  # number of time points
    t_MN = min(time_f)  # min time
    print(len(time_f))
    dt = np.mean(np.diff(time_f))  # average of all differences of time points
    Fs = 1 / dt  # Sampling Frequency
    # print (Fs)
    Fn = Fs / 2  # Nyquist Frequency
    #
    # plt.plot (time,signal)
    # plt.grid (True)
    # plt.xlabel('Time')
    # plt.ylabel('Blood Pressure (mmHg)')
    # plt.title('Original Signal')
    # plt.show()

    # Own Moving Mean
    window_size = 100
    half_window = 100 / 2
    sig_len = len(signal1)
    sig_len_end = sig_len - 50
    print(sig_len_end)
    first50 = []
    beginning_sum = []
    middle = []
    last50 = []
    sum_for = []
    signal1 = signal1.tolist()

    for x in range(0, 50):
        sum_for = sum_for + [float(signal1[x])]
        sum1 = sum(sum_for)
        sum_lat = float(sum(signal1[x + 1:x + 50]))
        total_len = float(50 + x)
        total_sum = sum_lat + sum1
        # print (total_sum)
        average = (total_sum) / total_len
        first50 = first50 + [average]

    for x in range(50, sig_len_end):
        mid_sum = float(sum(signal1[x - 50:x + 50]))
        middle = middle + [float(mid_sum / 100)]

    for x in range(sig_len_end, sig_len):
        sums = float(sum(signal1[x - 50:sig_len]))
        length = 50 + sig_len - x
        average = sums / length
        last50 = last50 + [average]

    minus_signal = first50 + middle + last50
    sub_signal = np.array(signal1) - np.array(minus_signal)
    print(sub_signal)

    # Shortening the subtracted signal by removing the initial spike
    signal_max = max(sub_signal)
    signal_max_index = np.argmax(sub_signal)
    print(signal_max_index)
    sub_signal_end = len(sub_signal)
    sub_signal_new = sub_signal[signal_max_index:sub_signal_end]
    signal_zero_mask = sub_signal_new < 0

    sig_zero_indices = np.where(signal_zero_mask)[0]  # indexes from sys_mask where peaks are higher
    signal_zero = sub_signal_new[sig_zero_indices]  # peak values

    print(signal_zero)
    signal_zero = signal_zero.tolist()
    signal_zero_min = signal_zero[0]
    sig_zero_indices = sig_zero_indices.tolist()
    signal_zero_min_index = sig_zero_indices[0]
    threshold_max_index = signal_zero_min_index + signal_max_index + 1
    print(signal_zero_min)
    print(signal_zero_min_index)
    print(threshold_max_index)

    signal_new = sub_signal[threshold_max_index:sub_signal_end]
    print(len(signal_new))
    time_new = time_f[threshold_max_index:sub_signal_end]

    # Applying Moving Mean to Shortened signal
    # Own Moving Mean
    sig_len = len(signal_new)
    sig_len_end30 = sig_len - 14
    first30 = []
    beginning_sum = []
    middle = []
    last30 = []
    sum_for = []
    # x=[]
    # x=[0]
    # xarray=int_to_array(x)
    # signal=signal.tolist()

    for x in range(0, 14):
        sum_for = sum_for + [float(signal_new[x])]
        sum1 = sum(sum_for)
        sum_lat = float(sum(signal_new[x + 1:x + 7]))
        total_len = float(7 + x)
        total_sum = sum_lat + sum1
        # print (total_sum)
        average = (total_sum) / total_len
        first30 = first30 + [average]

    for x in range(14, sig_len_end):
        mid_sum = float(sum(signal_new[x - 7:x + 7]))
        middle = middle + [float(mid_sum / 14)]

    for x in range(sig_len_end, sig_len):
        sums = float(sum(signal_new[x - 14:sig_len]))
        length = 14 + sig_len - x
        average = sums / length
        last30 = last30 + [average]

    signal_mean1 = first30 + middle + last30
    signal_mean_abs = np.abs(signal_mean1)

    ws = 50
    hws = 25
    sig_len100 = len(signal_mean_abs)
    sig_len_end100 = sig_len100 - ws
    first100 = []
    beginning_sum = []
    middle100 = []
    last100 = []
    sum_for100 = []
    # x=[]
    # x=[0]
    # xarray=int_to_array(x)
    # signal=signal.tolist()

    for x in range(0, ws):
        sum_for100 = sum_for100 + [float(signal_mean_abs[x])]
        sum1100 = sum(sum_for100)
        sum_lat100 = float(sum(signal_mean_abs[x + 1:x + ws]))
        total_len100 = float(hws + x)
        total_sum100 = sum_lat100 + sum1100
        # print (total_sum)
        average100 = total_sum100 / total_len100
        first100 = first100 + [average100]

    for x in range(ws, sig_len_end100):
        mid_sum100 = float(sum(signal_mean_abs[x - hws:x + hws]))
        middle100 = middle100 + [float(mid_sum100 / ws)]

    for x in range(sig_len_end100, sig_len100):
        sums100 = float(sum(signal_mean_abs[x - hws:sig_len100]))
        length100 = hws + sig_len100 - x
        average100 = sums100 / length100
        last100 = last100 + [average100]

    signal_mean = first100 + middle100 + last100

    time_len = len(time_new)
    end_time = time_new[time_len - 1]
    time_mask = (time_new > 1.25) & (time_new < end_time - 1.25)
    print(time_mask)
    time_zero_indices = np.where(time_mask)[0]  # indexes from sys_mask where peaks are higher
    time_new1 = time_new[time_zero_indices]
    print(len(time_new1))
    print(time_new1)
    index_time_start = time_zero_indices[0]
    index_time_end = max(time_zero_indices)
    print(index_time_start)
    print(index_time_end)

    signal_mean_filter = signal_mean[index_time_start:index_time_end + 1]
    print(len(signal_mean_filter))
    time_start = time_new[threshold_max_index + index_time_start]
    print(time_start)

    # plt.plot(time_new, np.abs(signal_mean_abs))
    # plt.grid(True)
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')  # FIND units
    # plt.title('Moving Mean')
    # plt.show()

    signal_mf_overlay_mean = np.abs(signal_mean_filter)

    # Band-pass filter
    # Sampling rate, cutoff frequencies (butterworth filters)
    lowcut = 0.4  # Low cutoff frequency, Hz
    highcut = 3  # High cutoff frequency, Hz

    # Butterworth bandpass filter
    low = lowcut / Fn
    high = highcut / Fn
    [b, a] = butter(3, [low, high], btype='band')
    print(b)
    print(a)

    # Apply the filter to the amplitude data using filtfilt

    signal_filter = filtfilt(b, a, signal1)
    print(signal_filter)

    # # Plot the original and filtered signals
    # plt.figure(figsize=(10, 5))
    # plt.plot(time,signal, label='Original signal')
    # plt.plot(time ,signal_filter, label='Filtered signal')
    # plt.legend(loc='best')
    # #plt.ylim(0, 50000)
    # plt.grid(True)
    # plt.xlabel('time(s)')
    # plt.ylabel('pressure(mmHg)')
    # plt.title('Blood Pressure Recording')
    # plt.show()
    #
    # #zoomed in of filtered signal
    # plt.plot (time,signal_filter)
    # plt.grid (True)
    # plt.xlabel('time(s)')
    # plt.ylabel('pressure(mmHg)')
    # plt.title('Bandpass Filtered Blood Pressure')
    # plt.show()

    # FFT of full signal
    freq = np.fft.fftfreq(len(time_new1), np.mean(np.diff(time_new1)))  # frequency domain of signal
    print(freq)
    FT_signal_filter = fft(signal_mean_filter)  # fourier transform of bandpass filtered signal
    #
    # Plot of Fourier transform
    # plt.plot (freq,np.abs(FT_signal_filter))
    # plt.grid (True)
    # plt.xlabel('Frequency (in Hz)')
    # plt.ylabel('Amplitude') #FIND units
    # plt.title('Frequency Spectrum')
    # plt.show()

    # FFT of half of signal
    Nm = len(signal_mean_filter)  # number of data points collected from filtered signal
    Fvm = int(Nm / 2)  # dividing number of data points in half

    FT_signal_filter_half = FT_signal_filter.tolist()  # converting np.array into a list to use del function
    del FT_signal_filter_half[Fvm:Nm]  # remove half of BP data points
    half_len = len(FT_signal_filter_half)  # confirm only half data points are to be recorded
    half_freq = freq.tolist()
    del half_freq[Fvm:Nm]  # remove half of frequency/time data points

    FT_filter_pks = np.abs(FT_signal_filter_half)  # make all signal values positive
    #
    # #Plot half of signal
    # plt.plot (half_freq,np.abs(FT_signal_filter_half))
    # plt.grid (True)
    # plt.xlabel('Frequency (in Hz)')
    # plt.ylabel('Amplitude')
    # plt.title('Frequency Spectrum(Filtered)')
    # plt.show()

    # Find peaks
    indexesFT, _ = find_peaks(FT_filter_pks)  # find the index of all peaks/local maxes
    print(indexesFT)
    pksFT = FT_filter_pks[indexesFT]  # values of all peaks
    print(pksFT)

    # Find peak frequencies
    # knowing the index of the peaks, find the frequencies at the same indexes
    pks_len = len(indexesFT)  # number of peaks found
    pks_freq = []
    for x in range(0, pks_len):
        idx = indexesFT[x]
        print(idx)
        pks_freq = pks_freq + [float(half_freq[idx])]
    print(pks_freq)

    # Extract 3 most dominant frequencies
    finding_max = []
    finding_max = pksFT  # create new variable of peaks that can be altered to find the top 5 maxes
    top5_vals = []  # top 5 maxes
    top5_idx = []  # index of top 5 maxes

    # max function can only find one max
    # this  loop finds the max, records the value and index
    # then, it replaces it with a 0, no longer making it a max
    # repeat the loop, finding new "highest max" and repeat cycle to find all 3 maxes
    for x in range(0, 3):
        top5_vals = top5_vals + [float(max(finding_max))]
        max_indice = np.argmax(finding_max)  # index on max array
        max_int = indexesFT[max_indice]  # finding true index from index array
        top5_idx = top5_idx + [int(max_int)]
        finding_max[max_indice] = 0
    print(top5_vals)
    print(top5_idx)

    # with the indexes of top 5 maxes recorded, find their corresponding frequencies
    freqs_intr = []
    for x in range(0, 3):
        idx = top5_idx[x]
        freqs_intr = freqs_intr + [float(half_freq[idx])]
    print(freqs_intr)

    freqs_mean = np.mean(freqs_intr)
    HR = 60 * freqs_mean
    print(HR)

    # #Plots of peaks and top 5 peaks
    # plt.plot (half_freq,np.abs(FT_signal_filter_half),'k')
    # plt.plot (freqs_intr, top5_vals, 'bo',label='top 5 peaks')
    # plt.plot (pks_freq,pksFT,'r*', label='peaks')
    # plt.grid (True)
    # plt.xlabel('Frequency (in Hz)')
    # plt.ylabel('Amplitude')
    # plt.title('FFT Peak Detection')
    # plt.show()
    #
    # #Plots of peaks and top 5 peaks
    # plt.plot (time,signal_mf_overlay_mean)
    # plt.grid (True)
    # #plt.xlim (0,40)
    # plt.xlabel('time(s)')
    # plt.ylabel('Amplitude')
    # plt.title('Band pass and Overlay')
    # plt.show()

    # find peaks and index of signal overlay

    signal_mf_overlay_mean = np.array(signal_mf_overlay_mean)
    print(f"SMTC: {signal_mf_overlay_mean}")
    print(f"SMFOM: {signal_mf_overlay_mean}")

    # # Plots of peaks and top 5 peaks
    # plt.plot(time_new, signal_mean)
    # plt.grid(True)
    # # plt.xlim (0,40)
    # plt.xlabel('time(s)')
    # plt.ylabel('Amplitude')
    # plt.title('Band pass and Overlay')
    # plt.show()

    indexes, _ = find_peaks(signal_mf_overlay_mean)
    pks = signal_mf_overlay_mean[indexes]
    print(pks)
    pks_max = max(pks)
    print(pks_max)

    # Setting threshold for what is considered for systolic and diastolic
    systolic_thres = 0.5 * pks_max
    diastolic_thres = 0.7 * pks_max

    # create masks that determine what peaks have a value higher than the thresholds
    # "mask"-true or false array that says when value is higher than threshold
    sys_mask = pks > systolic_thres
    dias_mask = pks > diastolic_thres

    # record the values and indices of the peaks with higher value than systolic threshold
    print(f"systolic mask: {sys_mask}")
    possible_sys_indices = np.where(sys_mask)[0]  # indexes from sys_mask where peaks are higher
    pks_f = pks[possible_sys_indices]  # peak values
    sys_indexes_f = indexes[possible_sys_indices]  # indexes from original signal where systolic peaks occurs
    print(possible_sys_indices)
    print(pks_f)
    print(f"Systolic indexes: {sys_indexes_f}")

    print(dias_mask)
    possible_dias_indices = np.where(dias_mask)[0]  # indexes from dias_mask where peaks are higher
    pks_f = pks[possible_dias_indices]  # peak values
    dias_indexes_f = indexes[possible_dias_indices]  # indexes from original signal where diastolic peaks occurs
    print(possible_dias_indices)
    print(pks_f)
    print(dias_indexes_f)

    pks_index_min = min(sys_indexes_f)  # systolic will be at the first peak that passed systolic threshold
    print(pks_index_min)
    pks_index_max = max(dias_indexes_f)  # diastolic will be at the last peak that passed diastolic threshold
    print(pks_index_max)

    # Systolic and Diastolic Readings
    systolic = signal1[pks_index_min + threshold_max_index + index_time_start]
    print('SYSTOLIC:')
    print(systolic)
    diastolic = signal1[pks_index_max + threshold_max_index + index_time_start]
    print('DIASTOLIC:')
    print(diastolic)
    print('Heart Rate:')
    print(HR)

    delta = systolic - 130
    delta2 = 90 - systolic
    return delta, delta2, systolic, diastolic, HR

BTN_STYLE = {
    "bg": "#007acc",
    "fg": "white",
    "activebackground": "#005f99",
    "font": ("Helvetica", 14),
    "width": 20,
    "padx": 10,
    "pady": 5
}

ENTRY_STYLE = {
    "font": ("Helvetica", 14),
    "width": 30,
    "justify": "center"
}

LABEL_STYLE = {
    "font": ("Helvetica", 16),
    "pady": 10
}

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Capstone GUI")

        self.password_list = [["nadia.kora", "098"]]
        self.actionlog = []
        self.current_user = None

        self.frames = {}
        for F in (WelcomePage, LoginPage, CreateAccountPage, AccessDataPage, EHRPage, EHRAccessPage, ChoosePatientPage,PredictionModel,Pump_Setup, PredictionModelExP):
            page_name = F.__name__
            frame = F(parent=self, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("WelcomePage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

    def add_user(self, username, password):
        self.password_list.append([username, password])

    def log_action(self, action):
        user = self.current_user if self.current_user else "Unknown"
        self.actionlog.append(f"[{user}] did {action} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        actionlogging = []
        actionlogging.append(f"[{user}] did {action} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        data_added_new = pandas.DataFrame(actionlogging, columns=["Record"])
        # Append to existing file
        data_added_new.to_csv(f"LoggingFile.csv", mode='a', header=False,
                              index=False)

        print(f"[LOG] {action}")

class CenteredFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def place_centered_widgets(self, widgets):
        container = tk.Frame(self)
        container.pack(expand=True)
        for widget in widgets:
            widget.pack(pady=8)

class WelcomePage(CenteredFrame):

    def __init__(self, parent, controller):
        super().__init__(parent)

        pil_image = Image.open(("MBotsLogo1.jpg"))
        pil_image = pil_image.resize((400, 250))
        tk_image = ImageTk.PhotoImage(pil_image)


        widgets = [
            tk.Label(self, image=tk_image),
            tk.Label(self, text="Welcome to the BPCheck", **LABEL_STYLE),
            tk.Label(self, text="Choose authorization option:", font=("Helvetica", 14)),
            tk.Button(self, text="Log in", command=lambda: controller.show_frame("LoginPage"), **BTN_STYLE),
            tk.Button(self, text="Create Account", command=lambda: controller.show_frame("CreateAccountPage"), **BTN_STYLE)
        ]
        self.place_centered_widgets(widgets)
        self.image_ref = tk_image

class LoginPage(CenteredFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.entry_un = tk.Entry(self, **ENTRY_STYLE)
        self.entry_ps = tk.Entry(self, show="*", **ENTRY_STYLE)
        self.entry_un.insert(0, "nadia.kora")
        self.entry_ps.insert(0, "098")
        widgets = [
            tk.Label(self, text="Use your username and password to log in", **LABEL_STYLE),
            tk.Label(self, text="Username:", font=("Helvetica", 14)),
            self.entry_un,
            tk.Label(self, text="Password:", font=("Helvetica", 14)),
            self.entry_ps,
            tk.Button(self, text="Log In", command=lambda: self.login_check(controller), **BTN_STYLE),
            tk.Button(self, text="Back", command=lambda: controller.show_frame("WelcomePage"), **BTN_STYLE)
        ]
        self.place_centered_widgets(widgets)

    def login_check(self, controller):
        controller.current_user = self.entry_un.get()
        password_check = self.entry_ps.get()
        controller.log_action("Username Input")
        controller.log_action("Password Input")
        for username, password in controller.password_list:
            if controller.current_user == username and password_check == password:
                controller.log_action("Access granted")
                controller.show_frame("AccessDataPage")
                return
        controller.log_action("Access denied")

class CreateAccountPage(CenteredFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.entry_nun = tk.Entry(self, **ENTRY_STYLE)
        self.entry_nps = tk.Entry(self, show="*", **ENTRY_STYLE)
        widgets = [
            tk.Label(self, text="Create username and password", **LABEL_STYLE),
            tk.Label(self, text="Username:", font=("Helvetica", 14)),
            self.entry_nun,
            tk.Label(self, text="Password:", font=("Helvetica", 14)),
            self.entry_nps,
            tk.Button(self, text="Create Account", command=lambda: self.create_account(controller), **BTN_STYLE),
            tk.Button(self, text="Back", command=lambda: controller.show_frame("WelcomePage"), **BTN_STYLE)
        ]
        self.place_centered_widgets(widgets)

    def create_account(self, controller):
        username_new = self.entry_nun.get()
        password_new = self.entry_nps.get()
        controller.add_user(username_new, password_new)
        controller.log_action("Account created")
        controller.show_frame("LoginPage")

class AccessDataPage(CenteredFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.p_name = tk.Entry(self, **ENTRY_STYLE)
        self.p_name.insert(0, "MiracleBots")
        widgets = [
            tk.Label(self, text="Patient Data Access", **LABEL_STYLE),
            tk.Label(self, text="Type patient's name", font=("Helvetica", 14)),
            self.p_name,
            tk.Button(self, text="Collect BP data", command=lambda: self.collect_and_go(controller), **BTN_STYLE),
            tk.Button(self, text="Access Patient Data", command=lambda: self.access_EHR(controller), **BTN_STYLE),
            tk.Button(self, text="Setup the IV Pump", command=lambda: self.setup_pump(controller), **BTN_STYLE),
            tk.Button(self, text="Back", command=lambda: controller.show_frame("LoginPage"), **BTN_STYLE)
        ]
        self.place_centered_widgets(widgets)

    def setup_pump(selfself,controller):
        controller.show_frame("Pump_Setup")
        controller.log_action("Pump is being setup")

    def collect_and_go(self, controller):
        controller.log_action("Patient's Name Input")
        choose_page = controller.frames["ChoosePatientPage"]
        choose_page.patient_name = self.p_name.get()
        controller.show_frame("ChoosePatientPage")

    def access_EHR(self, controller):
        controller.frames["EHRAccessPage"].retrieve_data(self.p_name.get())
        controller.show_frame("EHRAccessPage")
        controller.log_action("Access Patient Data")

class ChoosePatientPage(CenteredFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.patient_name = None
        self.collection_periodicity = tk.Entry(self, **ENTRY_STYLE)
        self.collection_periodicity.insert(0, "15")
        self.systolic_threshold_hyper = tk.Entry(self, **ENTRY_STYLE)
        self.systolic_threshold_hyper.insert(0, "130")
        self.systolic_threshold_hypo = tk.Entry(self, **ENTRY_STYLE)
        self.systolic_threshold_hypo.insert(0, "90")

        widgets = [
            tk.Label(self, text="Choose Patient Type", **LABEL_STYLE),
            tk.Button(self, text="New Patient", command=lambda: self.start_data_collection(controller), **BTN_STYLE),
            tk.Button(self, text="Existing Patient", command=lambda: self.add_new_data(controller), **BTN_STYLE),
            tk.Label(self, text="Collection Frequency (minutes)", font=("Helvetica", 14)),
            self.collection_periodicity,
            tk.Label(self, text="Systolic Threshold for Hypertension", font=("Helvetica", 14)),
            self.systolic_threshold_hyper,
            tk.Label(self, text="Systolic Threshold for Hypotension", font=("Helvetica", 14)),
            self.systolic_threshold_hypo,
            tk.Button(self, text="Back", command=lambda: controller.show_frame("AccessDataPage"), **BTN_STYLE)

        ]
        self.place_centered_widgets(widgets)


    def start_data_collection(self, controller):
        self.choose_patient_type(controller, new_patient=True)

    def add_new_data(self, controller):
        self.choose_patient_type(controller, new_patient=False)

    def choose_patient_type(self, controller, new_patient=True):
        periodicity_str = self.collection_periodicity.get()
        periodicity = int(periodicity_str)
        systolic_threshold_hyper_str = self.systolic_threshold_hyper.get()
        systolic_threshold_hyper_int = int(systolic_threshold_hyper_str)
        systolic_threshold_hypo_str = self.systolic_threshold_hypo.get()
        systolic_threshold_hypo_int = int(systolic_threshold_hypo_str)
        controller.frames["EHRPage"].type_new_patient = new_patient
        controller.frames["EHRPage"].start_data_collection(controller, self.patient_name, periodicity,systolic_threshold_hyper_int,systolic_threshold_hypo_int)
        controller.show_frame("EHRPage")
        controller.log_action("Patient type chosen")

class Pump_Setup(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        tk.Button(
            self,
            text="Air bubble prevention",
            command=lambda: threading.Thread(target=lambda: asyncio.run(self.send_ble_for_air_removal())).start(),
            **BTN_STYLE).pack(pady=10)

        tk.Button(self,
            text="Calibrate pump flow",
            command=self.calibrate_pump,
            **BTN_STYLE
        ).pack(pady=10)

        tk.Button(
            self,
            text="Back",
            command=lambda: controller.show_frame("AccessDataPage"),
            **BTN_STYLE
        ).pack(pady=10)

    async def send_ble_for_air_removal(self):
        device_2 = await BleakScanner.find_device_by_name("ESP32_LED", timeout=10.0)
        if device_2:
            async with BleakClient(device_2) as client_2:
                input_dosage = 500
                duration = 10
                self.controller.log_action("Connected to ESP32-2")
                try:
                    rpm = float(input_dosage) / 8 * 1000
                    bytes_rpm = int(rpm).to_bytes(2, byteorder='little')
                    bytes_duration = int(duration).to_bytes(2, byteorder='little')
                    bytes_drug_type = int(1).to_bytes(1, byteorder='little')
                    await client_2.write_gatt_char("abcd1001-4321-6789-4321-fedcba987654",
                                                   bytes_rpm + bytes_duration + bytes_drug_type)
                    self.controller.log_action("Dosage Sent to ESP32-2")
                except:
                    self.controller.log_action("ESP32-2 not found")

    def calibrate_pump(self):
            window = tk.Toplevel()
            window.title("Calibrate Pump")

            tk.Label(window, text="Place a collection vessel on the scale", **LABEL_STYLE).pack()
            tk.Label(window, text="Click Start when ready", **LABEL_STYLE).pack()

            tk.Button(window,text="Start Calibration",command=lambda: threading.Thread(target=lambda: asyncio.run(self.run_calibration(window))).start(),**BTN_STYLE).pack(pady=10)

    async def run_calibration(self, window):
        await self.send_ble_for_calibration()
        time.sleep(120)  # or use asyncio.sleep(120) inside an async function
        window.destroy()
        self.prompt_mass_entry()

    async def send_ble_for_calibration(self):
        device_2 = await BleakScanner.find_device_by_name("ESP32_LED", timeout=10.0)
        input_dosage = 150
        duration = 2
        if device_2:
            async with BleakClient(device_2) as client_2:
                self.controller.log_action("Connected to ESP32-2")
                try:
                    rpm = float(input_dosage) / 8 * 1000
                    bytes_rpm = int(rpm).to_bytes(2, byteorder='little')
                    bytes_duration = int(duration).to_bytes(2, byteorder='little')
                    bytes_drug_type = int(1).to_bytes(1, byteorder='little')
                    await client_2.write_gatt_char("abcd1001-4321-6789-4321-fedcba987654",
                                                   bytes_rpm + bytes_duration + bytes_drug_type)
                    self.controller.log_action("Dosage Sent to ESP32-2")
                except:
                    self.controller.log_action("ESP32-2 not found")

    def prompt_mass_entry(self):
        window = tk.Toplevel()
        window.title("Enter Weight")

        tk.Label(window, text="Enter mass collected (grams):", **LABEL_STYLE).pack()
        entry = tk.Entry(window, **ENTRY_STYLE)
        entry.pack()

        def compute_ratio():
            try:
                grams = float(entry.get())
                ratio = grams * 30 / 150 * 8
                print(f"Calibration Ratio: {ratio:.2f}")
            except ValueError:
                print("Invalid number")

        tk.Button(window, text="Submit", command=compute_ratio, **BTN_STYLE).pack(pady=10)



class EHRAccessPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.plot_area = tk.Frame(self)
        self.plot_area.pack(expand=True, pady=20)
        tk.Button(self, text="Back", command=lambda: controller.show_frame("WelcomePage"), **BTN_STYLE)

    def retrieve_data(self, patient_name):
        data_past = pandas.read_csv(f"BPdata_{patient_name}.csv")
        # #columns = ["Date and Time", "Systolic blood pressure",
        #            "Diastolic Blood Pressure",
        #            "Heart rate (bpm)",
        #            "Drug type", "Recommended Dosage",
        #            "Administered Dosage"]
        #date_n_time = data_past["Date and Time"]
        date_n_time = pandas.to_datetime(data_past["Date and Time"])
        minutes_since_start = (date_n_time - date_n_time[0]).dt.total_seconds() / 60

        print(date_n_time)
        print(minutes_since_start)
        systolic_extr = data_past["Systolic blood pressure"]
        diastolic_extr = data_past["Diastolic Blood Pressure"]
        HR_extr = data_past["Heart rate (bpm)"]
        drug_type_extr = data_past["Drug type"]
        admin_dosage_extr = pandas.to_numeric(data_past["Administered Dosage"])
        admin_duration_extr = pandas.to_numeric(data_past["Set Duration"])
        print(drug_type_extr)
        # y_min = min(y) - 5
        # y_max = max(y) + 5
        # x_min = min(x) - 1
        # x_max = max(x) + 1
        time_offset = np.arange(0, 3601, 1)

        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(minutes_since_start, systolic_extr, label="Systolic BP",marker='o',linewidth=2, color='blue')
        ax.plot(minutes_since_start, diastolic_extr, label="Diastolic BP",marker='o',linewidth=2, color='red')

        for i in range(len(drug_type_extr)):
            if drug_type_extr.iloc[i] == "Nicardipine":
                time_threshold = minutes_since_start.iloc[i]
                if i + 2 < len(minutes_since_start):
                    time_threshold2 = minutes_since_start.iloc[i + 2]
                else:
                    time_threshold2 = time_threshold + 5  # optional fallback

                time_offset_i = np.arange(time_threshold * 60, time_threshold2 * 60, 1)  # still in seconds
                time_offset_min = np.arange(time_threshold, time_threshold2, 1 / 60)  # in minutes

                # 🛠 Shift to local time starting at 0
                time_offset_local = time_offset_i - time_offset_i[0]

                new_dosage = admin_dosage_extr.iloc[i]
                Delta_BP = 1.7 * (6.9249 * math.log(float(new_dosage)) + 24.9)
                tau_decrease_high = 70 / math.log(10)
                tau_recovery_high = 4 * 60 / math.log(10)
                Delta_BP_recovery = 0.66 * Delta_BP

                print(f"Systolic at dose {i}: {systolic_extr.iloc[i]}")
                print(f"time_offset_local (first 10): {time_offset_local[:10]}")

                modeled_BP_sys = systolic_extr.iloc[i] - Delta_BP * (
                            1 - np.exp(-time_offset_local / tau_decrease_high)) + \
                                 Delta_BP_recovery * (1 - np.exp(-time_offset_local / tau_recovery_high))

                modeled_BP_dias = diastolic_extr.iloc[i] - Delta_BP * (1 - np.exp(-time_offset_local / tau_decrease_high)) + Delta_BP_recovery * (1 - np.exp(-time_offset_local / tau_recovery_high))

                print(f"modeled_BP_sys (first 10): {modeled_BP_sys[:10]}")

                # ax.plot(time_threshold, label=f"{drug_type_extr} of {admin_dosage_extr}", linestyle='--')
                ax.plot(time_offset_min, modeled_BP_sys,linestyle='--', label="Systolic BP Response to Nicardipine Administration", linewidth=2, color='blue')
                ax.plot(time_offset_min, modeled_BP_dias,linestyle='--', label="Diastolic BP Response to Nicardipine Administration", linewidth=2, color='red')


            if drug_type_extr.iloc[i] == "IV Fluids":
                time_threshold = minutes_since_start.iloc[i]
                if i + 1 < len(minutes_since_start):
                    time_threshold2 = minutes_since_start.iloc[i + 1]
                else:
                    time_threshold2 = time_threshold + 5  # optional fallback

                time_offset_i = np.arange(time_threshold * 60, time_threshold2 * 60, 1)  # still in seconds
                time_offset_min = np.arange(time_threshold, time_threshold2, 1 / 60)  # in minutes

                # 🛠 Shift to local time starting at 0
                time_offset_local = time_offset_min - time_offset_min[0]

                new_dosage = admin_dosage_extr.iloc[i]
                D = admin_duration_extr.iloc[i]
                t = np.arange(time_threshold, time_threshold+D, 1 / 60)
                t_local = np.arange(0, D, 1 / 60)
                t2_local = np.arange(D,D+60,1/60)
                t2 = (time_threshold+D,60+time_threshold,1 / 60)

                A = 0.7 * ((-0.3333 * D + 28) + 2)
                k_rise = (D - 75) / (-750)
                k_decay = 0.0002 * D + 0.016
                systolic_rise = systolic_extr.iloc[i] + A * (1 - np.exp(-k_rise * t_local))

                systolic_peak = systolic_rise[-1]

                systolic_decay = systolic_peak * np.exp(-k_decay * (t2_local - D)) + systolic_extr.iloc[i] * (1 - np.exp(-k_decay * (t2_local - D)))


                diastolic_rise = diastolic_extr.iloc[i] + A * (1 - np.exp(-k_rise * t_local))

                diastolic_peak = diastolic_rise[-1]

                diastolic_decay = diastolic_peak * np.exp(-k_decay * (t2_local - D)) + diastolic_extr.iloc[i] * (
                            1 - np.exp(-k_decay * (t2_local - D)))

                systolic_curve = np.concatenate([systolic_rise, systolic_decay])
                diastolic_curve = np.concatenate([diastolic_rise, diastolic_decay])
                time_offset2 = np.concatenate([t_local + time_threshold, t2_local + time_threshold])
                print(f"Systolic for IV: {systolic_curve}")


                # ax.plot(time_threshold, label=f"{drug_type_extr} of {admin_dosage_extr}", linestyle='--')
                ax.plot(time_offset2, systolic_curve, linestyle='--', label="Systolic BP Response to IV Fluids Bolus", linewidth=2, color='blue')
                ax.plot(time_offset2, diastolic_curve, linestyle='--', label="Diastolic BP Response to IV Fluids Bolus", linewidth=2, color='red')

        ax.legend()
        ax.set_xlabel("Time (minutes)", fontsize=14, fontname="Helvetica")
        ax.set_ylabel("Blood pressure (mmHg)", fontsize=14, fontname="Helvetica")

        # #ax.set_ylim(y_min, y_max)
        # ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_area)
        canvas.draw()
        canvas.get_tk_widget().pack()


class PredictionModel(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.plot_area = tk.Frame(self)
        self.plot_area.pack(expand=True, pady=20)
        controller.log_action("Start Modeling Response for New Patient")

    def model_response(self, new_iv_duration_speed, new_dosage, systolic, diastolic, nicardipine_admin):
        time_offset = np.arange(0, 3601, 1)
        for widget in self.plot_area.winfo_children():
            widget.destroy()
        if nicardipine_admin == 2:
            Delta_BP = 1.7 * (6.9249 * math.log(float(new_dosage)) + 24.9)
            tau_decrease_high = 70 / math.log(10)
            tau_recovery_high = 4 * 60 / math.log(10)
            Delta_BP_recovery = 0.66 * Delta_BP

            modeled_BP_sys = systolic - Delta_BP * (1 - np.exp(-time_offset / tau_decrease_high)) + \
                              Delta_BP_recovery * (1 - np.exp(-time_offset / tau_recovery_high))
            modeled_BP_dias = diastolic - Delta_BP * (1 - np.exp(-time_offset / tau_decrease_high)) + \
                               Delta_BP_recovery * (1 - np.exp(-time_offset / tau_recovery_high))

            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(time_offset, modeled_BP_sys, label="Systolic")
            ax.plot(time_offset, modeled_BP_dias, label="Diastolic")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Predicted BP (mmHg)")
            ax.legend()

            canvas = FigureCanvasTkAgg(fig, master=self.plot_area)
            canvas.draw()
            canvas.get_tk_widget().pack()
        elif nicardipine_admin == 1:

            D = new_iv_duration_speed
            print(F"Duration: {D}")
            t_local = np.arange(0, D, 1 / 60)
            t2_local = np.arange(D, D + 60, 1 / 60)

            A = 0.7 * ((-0.3333 * D + 28) + 2)
            k_rise = (D - 75) / (-750)
            k_decay = 0.0002 * D + 0.016
            systolic_rise = systolic + A * (1 - np.exp(-k_rise * t_local))

            systolic_peak = systolic_rise[-1]

            systolic_decay = systolic_peak * np.exp(-k_decay * (t2_local - D)) + systolic * (
                        1 - np.exp(-k_decay * (t2_local - D)))

            diastolic_rise = diastolic + A * (1 - np.exp(-k_rise * t_local))

            diastolic_peak = diastolic_rise[-1]

            diastolic_decay = diastolic_peak * np.exp(-k_decay * (t2_local - D)) + diastolic * (
                    1 - np.exp(-k_decay * (t2_local - D)))

            systolic_curve = np.concatenate([systolic_rise, systolic_decay])
            diastolic_curve = np.concatenate([diastolic_rise, diastolic_decay])
            time_offset2 = np.concatenate([t_local, t2_local])
            print(f"Systolic for IV: {systolic_curve}")

            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(time_offset2, systolic_curve, label="Systolic")
            ax.plot(time_offset2, diastolic_curve, label="Diastolic")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Predicted BP after IV Fluid Administration (mmHg)")
            ax.legend()

            canvas = FigureCanvasTkAgg(fig, master=self.plot_area)
            canvas.draw()
            canvas.get_tk_widget().pack()

class PredictionModelExP(tk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.plot_area = tk.Frame(self)
        self.plot_area.pack(expand=True, pady=20)
        tk.Label(self, text="Modelling 1", **LABEL_STYLE)
        controller.log_action("Start Modeling Response for Existing Patient")

    def model_response_exp(self, new_dosage, new_iv_duration_speed, systolic, diastolic, nicardipine_admin,
                           time_recorded, patient_name):
        print(f"Entered model_response_exp with dosage={new_dosage}, time={time_recorded}, name={patient_name}")

        # Clear old content first
        for widget in self.plot_area.winfo_children():
            widget.destroy()

        # Show plotting status
        tk.Label(self.plot_area, text="Plotting in progress...", bg="lightyellow").pack()

        time_offset = np.arange(0, 3601, 1)

        data_past_m = pandas.read_csv(f"BPdata_{patient_name}.csv")
        # #columns = ["Date and Time", "Systolic blood pressure",
        #            "Diastolic Blood Pressure",
        #            "Heart rate (bpm)",
        #            "Drug type", "Recommended Dosage",
        #            "Administered Dosage"]
        date_n_time = pandas.to_datetime(data_past_m["Date and Time"])
        minutes_since_start = (date_n_time - date_n_time[0]).dt.total_seconds() / 60

        print(date_n_time)

        systolic_extr = data_past_m["Systolic blood pressure"]
        diastolic_extr = data_past_m["Diastolic Blood Pressure"]
        HR_extr = data_past_m["Heart rate (bpm)"]
        drug_type_extr = data_past_m["Drug type"]
        admin_dosage_extr = pandas.to_numeric(data_past_m["Administered Dosage"])
        admin_duration_extr = pandas.to_numeric(data_past_m["Set Duration"])
        print(drug_type_extr)
        # y_min = min(y) - 5
        # y_max = max(y) + 5
        # x_min = min(x) - 1
        # x_max = max(x) + 1

        minutes_recorded = (pandas.to_datetime(time_recorded) - date_n_time[0]).total_seconds() / 60

        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)

        ax.plot(minutes_since_start, systolic_extr, label="Systolic BP")
        ax.plot(minutes_since_start, diastolic_extr, label="Diastolic BP")

        for i in range(len(drug_type_extr)):
            if drug_type_extr.iloc[i] == "Nicardipine":
                time_threshold = minutes_since_start.iloc[i]
                if i + 2 < len(minutes_since_start):
                    time_threshold2 = minutes_since_start.iloc[i + 2]
                else:
                    time_threshold2 = time_threshold + 5  # optional fallback

                time_offset_i = np.arange(time_threshold * 60, time_threshold2 * 60, 1)  # still in seconds
                time_offset_min = np.arange(time_threshold, time_threshold2, 1 / 60)  # in minutes

                # 🛠 Shift to local time starting at 0
                time_offset_local = time_offset_i - time_offset_i[0]

                new_dosage = admin_dosage_extr.iloc[i]
                Delta_BP = 1.7 * (6.9249 * math.log(float(new_dosage)) + 24.9)
                tau_decrease_high = 70 / math.log(10)
                tau_recovery_high = 4 * 60 / math.log(10)
                Delta_BP_recovery = 0.66 * Delta_BP

                print(f"Systolic at dose {i}: {systolic_extr.iloc[i]}")
                print(f"time_offset_local (first 10): {time_offset_local[:10]}")

                modeled_BP_sys = systolic_extr.iloc[i] - Delta_BP * (
                        1 - np.exp(-time_offset_local / tau_decrease_high)) + \
                                 Delta_BP_recovery * (1 - np.exp(-time_offset_local / tau_recovery_high))

                modeled_BP_dias = diastolic_extr.iloc[i] - Delta_BP * (
                            1 - np.exp(-time_offset_local / tau_decrease_high)) + Delta_BP_recovery * (
                                              1 - np.exp(-time_offset_local / tau_recovery_high))

                print(f"modeled_BP_sys (first 10): {modeled_BP_sys[:10]}")

                # ax.plot(time_threshold, label=f"{drug_type_extr} of {admin_dosage_extr}", linestyle='--')
                ax.plot(time_offset_min, modeled_BP_sys, linestyle='--',
                        label="Systolic BP Response to Nicardipine Administration", linewidth=2, color='blue')
                ax.plot(time_offset_min, modeled_BP_dias, linestyle='--',
                        label="Diastolic BP Response to Nicardipine Administration", linewidth=2, color='red')

            if drug_type_extr.iloc[i] == "IV Fluids":
                time_threshold = minutes_since_start.iloc[i]
                if i + 1 < len(minutes_since_start):
                    time_threshold2 = minutes_since_start.iloc[i + 1]
                else:
                    time_threshold2 = time_threshold + 5  # optional fallback

                time_offset_i = np.arange(time_threshold * 60, time_threshold2 * 60, 1)  # still in seconds
                time_offset_min = np.arange(time_threshold, time_threshold2, 1 / 60)  # in minutes

                # 🛠 Shift to local time starting at 0
                time_offset_local = time_offset_min - time_offset_min[0]

                D = admin_duration_extr.iloc[i]
                t = np.arange(time_threshold, time_threshold + D, 1 / 60)
                t_local = np.arange(0, D, 1 / 60)
                t2_local = np.arange(D, D + 60, 1 / 60)
                t2 = (time_threshold + D, 60 + time_threshold, 1 / 60)

                A = 0.7 * ((-0.3333 * D + 28) + 2)
                k_rise = (D - 75) / (-750)
                k_decay = 0.0002 * D + 0.016
                systolic_rise = systolic_extr.iloc[i] + A * (1 - np.exp(-k_rise * t_local))

                systolic_peak = systolic_rise[-1]

                systolic_decay = systolic_peak * np.exp(-k_decay * (t2_local - D)) + systolic_extr.iloc[i] * (
                            1 - np.exp(-k_decay * (t2_local - D)))

                diastolic_rise = diastolic_extr.iloc[i] + A * (1 - np.exp(-k_rise * t_local))

                diastolic_peak = diastolic_rise[-1]

                diastolic_decay = diastolic_peak * np.exp(-k_decay * (t2_local - D)) + diastolic_extr.iloc[i] * (
                        1 - np.exp(-k_decay * (t2_local - D)))

                systolic_curve = np.concatenate([systolic_rise, systolic_decay])
                diastolic_curve = np.concatenate([diastolic_rise, diastolic_decay])
                time_offset2 = np.concatenate([t_local + time_threshold, t2_local + time_threshold])
                print(f"Systolic for IV: {systolic_curve}")

                # ax.plot(time_threshold, label=f"{drug_type_extr} of {admin_dosage_extr}", linestyle='--')
                ax.plot(time_offset2, systolic_curve, linestyle='--', label="Systolic BP Response to IV Fluids Bolus",
                        linewidth=2, color='blue')
                ax.plot(time_offset2, diastolic_curve, linestyle='--', label="Diastolic BP Response to IV Fluids Bolus",
                        linewidth=2, color='red')

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Predicted BP (mmHg)")
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_area)
        canvas.draw()
        canvas.get_tk_widget().pack()

class EHRPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.plot_area = tk.Frame(self)
        self.plot_area.pack(expand=True, pady=20)
        self.type_new_patient = None

    def add_new_data(self):
        type_new_patient = True

    def start_data_collection(self, controller, patient_name,periodicity,systolic_threshold_hyper_int,systolic_threshold_hypo_int):
        # Clear previous plots
        for widget in self.plot_area.winfo_children():
            widget.destroy()
        print(f"{systolic_threshold_hyper_int}/{systolic_threshold_hypo_int}")
        # Local state containers
        data_values = []
        time_stamps = []
        map_vals = []
        delta_map = []
        delta_map2 = []
        dosage_drug = []
        drug = []
        dosage_sent = False
        #rpm = 100
        sys_threshold = 160
        dias_threshold = 50

        ESP1_SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
        DATA_CHAR_UUID = "abcd0001-1234-5678-1234-56789abcdef0"
        READY_CHAR_UUID = "abcd0002-1234-5678-1234-56789abcdef0"

        ESP2_SERVICE_UUID = "87654321-4321-6789-4321-fedcba987654"
        WRITE_CHAR_UUID = "abcd1001-4321-6789-4321-fedcba987654"

        stop_event = asyncio.Event()
        time_start = time.time()

        def handle_notification(sender, data):
            value = int.from_bytes(data, byteorder='little')
            timestamp = time.time() - time_start

            if value == 0xFFFF:
                print("Threshold reached")
                stop_event_loop()
            else:
                data_values.append(value/10)
                time_stamps.append(timestamp)

        def stop_event_loop():
            # Schedule stop_event.set() safely
            loop = asyncio.get_event_loop()
            if not stop_event.is_set():
                loop.call_soon_threadsafe(stop_event.set)

        async def send_ble(input_dosage, duration, drug_type):
            device_2 = await BleakScanner.find_device_by_name("ESP32_LED", timeout=10.0)
            if device_2:
                async with BleakClient(device_2) as client_2:
                    self.controller.log_action("Connected to ESP32-2")

                    if input_dosage > 0:
                        print(f"Duration: {duration}")

                        rpm = float(input_dosage) / 8 * 100
                        print(f"RPM: {rpm}")
                        bytes_rpm = int(rpm).to_bytes(2, byteorder='little')
                        bytes_duration = int(duration).to_bytes(2, byteorder='little')
                        bytes_drug_type = int(drug_type).to_bytes(1, byteorder='little')
                        await client_2.write_gatt_char(WRITE_CHAR_UUID, bytes_rpm + bytes_duration + bytes_drug_type)
                        self.controller.log_action("Dosage Sent to ESP32-2")
            # else:
            #     self.controller.log_action("ESP32-2 not found")

        def dosage_administration(new_dosage, new_iv_duration_speed, drug_type, periodicity, systolic, diastolic, HR):
            print(f"SHOWING: {'PredictionModel' if self.type_new_patient else 'PredictionModelExP'}")
            new_window = tk.Toplevel()
            new_window.title("Drug Recommendation")

            if drug_type == 2:
                drug = "Nicardipine"
                nicardipine_admin = 2
                tk.Label(new_window, text=f"BP: {systolic}/{diastolic}. HR: {round(HR)}", **LABEL_STYLE).pack()
                tk.Label(new_window,
                         text=f"Suggested Dosage of nicardipine: {new_dosage * 6}mg/hour or {new_dosage * 10 * 6}mL/hour",
                         **LABEL_STYLE).pack()
                entry_dosage = tk.Entry(new_window, font=("Helvetica", 14))
                entry_dosage.insert(0, str(new_dosage * 6*10))
                entry_dosage.pack()
                tk.Label(new_window, text="mL/hour", **LABEL_STYLE).pack()
                entry_duration = tk.Entry(new_window, font=("Helvetica", 14))
                entry_duration.insert(0, "60")
                entry_duration.pack()
                tk.Label(new_window, text="minutes", **LABEL_STYLE).pack()

            elif drug_type == 1:
                drug = "IV fluids"
                nicardipine_admin = 1
                tk.Label(new_window, text=f"BP: {systolic}/{diastolic}. HR: {round(HR)}", **LABEL_STYLE).pack()
                tk.Label(new_window, text=f"Suggested Duration: {new_iv_duration_speed} minutes", **LABEL_STYLE).pack()
                entry_dosage = tk.Entry(new_window, font=("Helvetica", 14))
                entry_dosage.insert(0, "2100")
                entry_dosage.pack()
                tk.Label(new_window, text="30 mL/kg (70kg) or 2100 mL over", **LABEL_STYLE).pack()
                entry_duration = tk.Entry(new_window, font=("Helvetica", 14))
                entry_duration.insert(0, str(new_iv_duration_speed))
                entry_duration.pack()
                tk.Label(new_window, text="minutes", **LABEL_STYLE).pack()

            else:
                drug = "None"
                nicardipine_admin = 0
                tk.Label(new_window, text=f"BP: {systolic}/{diastolic}. HR: {round(HR)}", **LABEL_STYLE).pack()
                tk.Label(new_window, text="Patient's BP is in the set range", **LABEL_STYLE).pack()
                return  # No need to create a button or proceed further

            def on_administer():
                IV_or_n_duration = int(entry_duration.get())
                if drug_type == 2:
                    IV_or_n_dosage = float(entry_dosage.get())
                elif drug_type == 1:
                    IV_or_n_dosage = float(entry_dosage.get())/IV_or_n_duration * 60

                time_recorded = time.strftime('%Y-%m-%d %H:%M:%S')
                drug_label = "Nicardipine" if drug_type == 2 else "IV Fluids"

                data_saved = [[time_recorded, systolic, diastolic, HR, drug_label, new_dosage,
                               IV_or_n_dosage, new_iv_duration_speed, IV_or_n_duration]]

                df_columns = ["Date and Time", "Systolic blood pressure", "Diastolic Blood Pressure",
                              "Heart rate (bpm)",
                              "Drug type", "Recommended Dosage", "Administered Dosage", "Recommended Duration",
                              "Set Duration"]

                if self.type_new_patient:
                    pandas.DataFrame(data_saved, columns=df_columns).to_csv(f"BPdata_{patient_name}.csv", index=False)
                    controller.frames["PredictionModel"].model_response(IV_or_n_duration, IV_or_n_dosage, systolic,
                                                                        diastolic, nicardipine_admin)
                    controller.show_frame("PredictionModel")
                else:
                    pandas.DataFrame(data_saved, columns=df_columns).to_csv(f"BPdata_{patient_name}.csv", mode='a',
                                                                            header=False, index=False)
                    controller.frames["PredictionModelExP"].model_response_exp(IV_or_n_duration, IV_or_n_dosage,
                                                                               systolic, diastolic, nicardipine_admin,
                                                                               time_recorded, patient_name)
                    controller.show_frame("PredictionModelExP")

                threading.Thread(
                    target=lambda: asyncio.run(send_ble(IV_or_n_dosage, IV_or_n_duration, drug_type))).start()
                self.controller.log_action("Connecting to ESP32")

            tk.Button(new_window, text="Administer the dosage", command=on_administer, **BTN_STYLE).pack()

        async def main():
            nonlocal dosage_sent

            while True:
                data_values.clear()
                time_stamps.clear()
                self.controller.log_action("Scanning for ESP32-POT")
                stop_event.clear()

                # --- ESP32-1 ---
                device_1 = await BleakScanner.find_device_by_name("ESP32_POT")
                async with BleakClient(device_1) as client_1:
                    # Send thresholds
                    bytes_sys = int(sys_threshold).to_bytes(2, byteorder='little')
                    bytes_dias = int(dias_threshold).to_bytes(2, byteorder='little')
                    await client_1.write_gatt_char(DATA_CHAR_UUID, bytes_sys + bytes_dias)

                    # Start notifications
                    await client_1.start_notify(READY_CHAR_UUID, handle_notification)
                    self.controller.log_action("Waiting for data from ESP32-POT")
                    await stop_event.wait()
                    await client_1.stop_notify(READY_CHAR_UUID)

                # Calculate average MAP value
                if data_values:
                    delta, delta2, systolic, diastolic, HR = process_bp_signal(data_values, time_stamps)
                    controller.log_action(f"Systolic: {systolic}")
                    controller.log_action(f"Diastolic: {diastolic}")
                    controller.log_action(f"HR: {HR}")
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    timestamp2 = time.time() - time_start
                    map_vals.append(delta)
                    controller.log_action(f"Hypertensive delta is calculated as {delta}")
                    controller.log_action(f"Hypotensive delta2 is calculated as {delta2}")

                    # --- Data saving ---
                    data = [[time_stamps, data_values]]
                    df = pandas.DataFrame(data, columns=["Timestamp", "Button State"])
                    df.to_csv(f"PressureSensordata_{patient_name}.csv", index=False)

                    # Plot results

                    data_plot = [[t, v] for t, v in zip(time_stamps, data_values)]
                    x = [row[0] for row in data_plot]
                    y = [row[1] for row in data_plot]
                    y_min = min(y) - 5
                    y_max = max(y) + 5

                    def plot_result():
                        window = tk.Toplevel()
                        window.title("Plot Pressure Sensor")
                        fig = Figure(figsize=(5, 4), dpi=100)
                        ax = fig.add_subplot(111)
                        ax.plot(x, y)
                        ax.set_xlabel("Time (seconds)")
                        ax.set_ylabel("Pressure Sensor Readings (mmHg)")
                        ax.set_ylim(y_min, y_max)
                        ax.set_yticks(range(int(y_min), int(y_max) + 1, 5))

                        canvas = FigureCanvasTkAgg(fig, master=window)
                        canvas.draw()
                        canvas.get_tk_widget().pack()

                    self.plot_area.after(0, plot_result)
                    controller.log_action("Data Collected, Accessing Data")

                    delta = systolic - systolic_threshold_hyper_int
                    delta2 = systolic_threshold_hypo_int - systolic
                    print(f"Deltas: {delta}/{delta2}")

                    # Calculate dosage
                    if delta > 0:
                        delta_map.append(delta)
                        dosage = math.exp((delta / 1.7 - 24.9) / 6.9249)
                        dosage_drug.append(dosage)
                        drug_type = 2
                        drug_type_copy = drug_type
                        if not dosage_sent:
                            f_dosage = float(dosage)
                            choices = [0.5, 1, 2, 3, 5, 7.5, 10, 12.5, 15]
                            new_dosage = min(choices, key=lambda x: abs(x - f_dosage))
                            self.plot_area.after(0, lambda: tk.Label(
                                self.plot_area, text=f"Nicardipine Dosage: {new_dosage} mg").pack())
                            dosage_sent = True
                            new_iv_duration_speed = 0
                            self.controller.log_action("Nicardipine Dosage Suggested")
                            self.controller.log_action(f"Waiting for user confirmation before sending to ESP32-2, drug type: {drug_type}")
                            self.plot_area.after(0,
                                                 lambda d=new_dosage, s=new_iv_duration_speed, t=drug_type_copy, p=periodicity,
                                                        sys=systolic, dia=diastolic, hr=HR:
                                                 dosage_administration(d, s, t, p, sys, dia, hr))
                            return

                    elif delta2 > 0:
                        delta_map2.append(delta)
                        A =  9.7816*math.log(delta2) - 14.083
                        iv_duration_speed = ((38.32-delta2)/0.496)
                        iv_duration_speed = round(((A/0.7)-30)/(-0.3333))
                        dosage_drug.append(iv_duration_speed)
                        drug_type = 1
                        drug_type_copy = drug_type
                        if not dosage_sent:
                            f_iv_duration_speed = float(iv_duration_speed)
                            choices = [20, 30, 45, 60]
                            new_iv_duration_speed = min(choices, key=lambda x: abs(x - f_iv_duration_speed))
                            self.plot_area.after(0, lambda: tk.Label(
                                self.plot_area, text=f"IV Fluids Duration of the set dosage administation (speed): {new_iv_duration_speed} mg").pack())
                            dosage_sent = True
                            new_dosage = 0
                            self.controller.log_action("Ringer's Dosage Suggested")
                            self.controller.log_action("Waiting for user confirmation before sending to ESP32-2")
                            self.plot_area.after(0,
                                                 lambda d=new_dosage, s=new_iv_duration_speed, t=drug_type_copy, p=periodicity,
                                                        sys=systolic, dia=diastolic, hr=HR:
                                                 dosage_administration(d, s, t, p, sys, dia, hr))
                            return

                    else:
                        self.controller.log_action("Patient's Blood Pressure in range")
                        new_dosage=0
                        new_iv_duration_speed=0
                        drug_type =0
                        drug_type_copy = drug_type

                        self.plot_area.after(0, lambda d=new_dosage, s=new_iv_duration_speed, t=drug_type_copy, p=periodicity,
                                                       sys=systolic, dia=diastolic, hr=HR:
                        dosage_administration(d, s, t, p, sys, dia, hr))

                await asyncio.sleep(periodicity * 60)

        #
        # except Exception as e:
        #     self.controller.log_action(f"Error in BLE collection: {e}")
        #     self.plot_area.after(0, lambda: tk.Label(self.plot_area, text=f"Error: {e}", fg="red").pack())

        def run_ble_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(main())
            loop.close()

        # Start BLE logic in background
        threading.Thread(target=run_ble_thread).start()

if __name__ == "__main__":
    app = App()
    app.mainloop()
