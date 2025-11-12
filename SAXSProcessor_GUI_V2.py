import sys
import os

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QFileDialog, QLabel, QComboBox, QLineEdit, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar, QSplitter
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

from SAXSProcessor import BatchSAXSDataProcessor, SAXSDataProcessor



class SAXSGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Batch SAXS Processor GUI")
        self.resize(1600, 1120)
        self.file_list = []
        self.mask_file = None
        self.df = None
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        # ------------------------------------------------------------------
        # 1) Fichiers / masque / paramètres instrument & binning
        # ------------------------------------------------------------------
        self.files_label = QLabel("No data files")
        main_layout.addWidget(self.files_label)

        self.select_files_btn = QPushButton("Select data files")
        self.select_files_btn.clicked.connect(self.select_files)
        main_layout.addWidget(self.select_files_btn)

        self.data2png_btn = QPushButton("Save detector images as png files")
        self.data2png_btn.clicked.connect(self.save2png)
        main_layout.addWidget(self.data2png_btn)

        self.mask_label = QLabel("No mask file")
        main_layout.addWidget(self.mask_label)

        self.select_mask_btn = QPushButton("Select mask file (optional)")
        self.select_mask_btn.clicked.connect(self.select_mask)
        main_layout.addWidget(self.select_mask_btn)

        self.instrument_box = QComboBox()
        self.instrument_box.addItems(["ID02", "SWING", "LGC"])
        main_layout.addWidget(QLabel("Instrument"))
        main_layout.addWidget(self.instrument_box)

        self.binning_input = QLineEdit("1")
        main_layout.addWidget(QLabel("Binning"))
        main_layout.addWidget(self.binning_input)

        # ------------------------------------------------------------------
        # 2) Profils radiaux (d)  VS  Profils azimutaux (S)
        # ------------------------------------------------------------------
        # ---- Bloc gauche : Profils radiaux & d ----
        radial_group = QGroupBox("Radial profiles – Determination of correlation distances d")
        radial_layout = QFormLayout(radial_group)

        self.nb_peaks_input = QLineEdit("2")
        radial_layout.addRow("Number of peaks :", self.nb_peaks_input)

        self.azimuth_input = QLineEdit("90")
        radial_layout.addRow("Mean azimuthal angle of integration sector (°) :", self.azimuth_input)

        self.width_input = QLineEdit("90")
        radial_layout.addRow("Width of integration sector (°) :", self.width_input)

        self.run_btnd = QPushButton("Determination of d")
        self.run_btnd.clicked.connect(self.run_processing_d)
        radial_layout.addRow(self.run_btnd)

        #----- Bloc du milieu: détermination des pentes
        slope_group = QGroupBox("Radial profiles – Determination of slopes between q1 and q2")
        slope_layout = QFormLayout(slope_group)

        self.q1_input = QLineEdit("0.01")
        slope_layout.addRow("q1 :", self.q1_input)

        self.q2_input = QLineEdit("0.1")
        slope_layout.addRow("q2 :", self.q2_input)

        self.run_btnslope = QPushButton("Determination of slope")
        self.run_btnslope.clicked.connect(self.run_processing_slope)
        slope_layout.addRow(self.run_btnslope)
        
        # ---- Bloc droit : Profils azimutaux & S ----
        azim_group = QGroupBox("Azimuthal Profile – determination of nematic order parameter S")
        azim_layout = QFormLayout(azim_group)

        self.threshold_input = QLineEdit("0.01")
        azim_layout.addRow("Threshold :", self.threshold_input)

        self.qvalues_input = QLineEdit("0.034, 0.068")
        azim_layout.addRow("q-values :", self.qvalues_input)

        self.run_btnS = QPushButton("Determination of S")
        self.run_btnS.clicked.connect(self.run_processing_S)
        azim_layout.addRow(self.run_btnS)

        # ---- Layout horizontal qui rassemble les deux blocs ----
        profiles_layout = QHBoxLayout()
        profiles_layout.addWidget(radial_group)
        profiles_layout.addWidget(slope_group)
        profiles_layout.addWidget(azim_group)
        
        main_layout.addLayout(profiles_layout)

        # ------------------------------------------------------------------
        # 3) Progression et bouton combiné S + d
        # ------------------------------------------------------------------
        self.progress = QProgressBar()
        self.progress.setValue(0)
        main_layout.addWidget(self.progress)

        self.run_btn = QPushButton("Determination of S and d")
        self.run_btn.clicked.connect(self.run_processing)
        main_layout.addWidget(self.run_btn)

        # ------------------------------------------------------------------
        # 4) Tableau + figure AVEC TOOLBAR
        # ------------------------------------------------------------------
        splitter = QSplitter(Qt.Vertical)
        self.results_table = QTableWidget()
        splitter.addWidget(self.results_table)

        # Widget conteneur pour le canvas et la toolbar
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        
        # Créer le canvas
        self.canvas = FigureCanvas(Figure(figsize=(10, 6)))
        
        # Créer la toolbar de navigation
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        
        # Ajouter la toolbar puis le canvas au layout
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        plot_layout.setContentsMargins(0, 0, 0, 0)  # Supprimer les marges
        
        # Ajouter le widget plot au splitter
        splitter.addWidget(plot_widget)
        main_layout.addWidget(splitter)

        # ------------------------------------------------------------------
        # 5) Sauvegarde
        # ------------------------------------------------------------------
        self.save_btn = QPushButton("Save results as CSV file")
        self.save_btn.clicked.connect(self.save_results)
        main_layout.addWidget(self.save_btn)

        # ------------------------------------------------------------------
        # 6) Appliquer le layout principal
        # ------------------------------------------------------------------
        self.setLayout(main_layout)

    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
                    self,
                    "Select SAXS data files",
                    r"T:\LPCNO\NCO\Manips\DATA_SAXS\StageNguyen2025\SWING\feco_serie",  # <- répertoire par défaut
                    #'/home-local/ratel-ra/Documents/STAGES/Nguyen_Truong_2025/Co12',
                    "Fichiers HDF5/EDF (*.h5 *.edf)")
        if files:
            self.file_list = files
            self.files_label.setText(f"{len(files)} selected files")

    def select_mask(self):
        mask, _ = QFileDialog.getOpenFileName(self, "Select mask file (.edf)", r"T:\LPCNO\NCO\Manips\DATA_SAXS\StageNguyen2025")
        if mask:
            self.mask_file = mask
            self.mask_label.setText(os.path.basename(mask))

    def init_SAXSProcessor(self,file):

        binning = int(self.binning_input.text())
        instrument = self.instrument_box.currentText()
        threshold = float(self.threshold_input.text())
        qvalues = np.array(list(map(float, self.qvalues_input.text().split(','))))
                
        processor = SAXSDataProcessor(
                        file=file,
                        instrument=instrument,
                        qvalues=qvalues,
                        binning=binning,
                        threshold=threshold,
                        mask=self.mask_file
                    )
        return processor

    def run_processing_S(self):
        try:
            if not self.file_list:
                QMessageBox.warning(self, "Error", "No data file selected.")
                return

            path = os.path.dirname(self.file_list[0])
            qvalues = np.array(list(map(float, self.qvalues_input.text().split(','))))
            self.progress.setMaximum(len(self.file_list))
            self.progress.setValue(0)
            results = []

            for idx, file in enumerate(self.file_list):
                try:
                    processor = self.init_SAXSProcessor(file)
                    result_dict = processor.compute_S()
                    sample = processor.samplename
                    B = processor.B
                    data_array = processor.data  # image SAXS

                    # distances de corrélation
                    #d= processor.compute_correlation_distance()

                    for q in qvalues:
                        val = result_dict[str(q)]
                        background= val[0]
                        
                        I=val[1]
                        position= val[2]
                        x0_S=val[3]
                        """
                        if position<0 and position <-20:
                            position+=180 # position must be between 0 and 180 (we only consider this range for S calculation)
                        if position >20:
                            position -=90 # center position around 0 so that 0<S<1
                        """
                        gamma =val[4]
                        eta=val[5]
                        slope = val[6]
                        #S= results[str(qvalue)][7]
                        #r_squared=results[str(qvalue)][8]

                        results.append([
                            os.path.basename(file),
                            sample,
                            B,
                            q,
                            background,     #y0
                            slope,          # slope for linear background
                            I,              # PV intensity
                            x0_S,           # peak position
                            gamma,          # parameter correlated to peak width
                            eta,            # mixing parameter
                            val[7],        # S
                            val[8],        # R² (S)
                            #d,            # correlation distance
                            data_array     # image SAXS (tableau numpy)
                        ])
                    self.plot_azimuthal_profile(processor)
                except Exception as e:
                    print(f"Erreur fichier {file} : {e}")
                self.progress.setValue(idx + 1)

            self.df = pd.DataFrame(results, columns=[
                "File Name",
                "Sample",
                "B (mT)",
                "q",
                "background y0",
                "background slope",
                "PV intensity",
                "PV position",
                "PV sigma",
                "PV eta (mix)",
                "S",
                "R² (S)",
                "SAXS data"
            ])
            self.display_results(self.df)

            QMessageBox.information(self, "Success", "Determination of nematic order parameter done. Inspect results.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"{e}")

    def display_results(self, df: pd.DataFrame):
        self.results_table.setRowCount(len(df))
        self.results_table.setColumnCount(len(df.columns))
        self.results_table.setHorizontalHeaderLabels(df.columns)

        for i, row in df.iterrows():
            for j, col in enumerate(df.columns):
                self.results_table.setItem(i, j, QTableWidgetItem(str(row[col])))

        self.results_table.resizeColumnsToContents()
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def plot_azimuthal_profile(self, processor: SAXSDataProcessor):
        self.canvas.figure.clf()
        ax = self.canvas.figure.add_subplot(111)
        for q in processor.qvalues:
            chi, I = processor.pyFAI_extract_azimprofiles(q)
            ax.plot(chi, I, label=f'q = {q:.3f}')
        ax.set_xlabel("Azimutal angle(°)")
        ax.set_ylabel("Intensity")
        ax.legend()
        ax.set_title(processor.samplename)
        
        # Mettre à jour la toolbar après le plot
        self.canvas.figure.tight_layout()
        self.toolbar.update()
        self.canvas.draw()
    
    def run_processing_d(self):
        try:
            if not self.file_list:
                QMessageBox.warning(self, "Error", "No data file selected.")
                return

            nb_peaks=int(self.nb_peaks_input.text())
            azimuth=float(self.azimuth_input.text())
            width=float(self.width_input.text())
            self.progress.setMaximum(len(self.file_list))
            self.progress.setValue(0)
            results = []

            for idx, file in enumerate(self.file_list):
                try:
                    processor = self.init_SAXSProcessor(file)
                    sample = processor.samplename
                    B = processor.B
                    data_array = processor.data  # image SAXS

                    # distances de corrélation
                    d= processor.compute_correlation_distance(azimuth=azimuth,width=width,nb_peaks=nb_peaks,caving=True)
                    ratio_array= [d[0]/dist for dist in d]
                    results.append([
                        os.path.basename(file),
                        sample,
                        B,                        
                        d,            # correlation distance
                        ratio_array, # distance ratio (reference= 1st peak)  
                        data_array     # image SAXS (tableau numpy)
                    ])
                    self.plot_radial_profile(processor)
                    # update self.q_values_input with 2pi/d
                    q_array=[2*np.pi/dist for dist in d]
                    
                    line2write=','.join(f"{item:.4f}" for item in q_array)
                    self.qvalues_input.setText(line2write)
                except Exception as e:
                    print(f"Erreur fichier {file} : {e}")
                self.progress.setValue(idx + 1)

            self.df = pd.DataFrame(results, columns=[
                "File Name",
                "Sample",
                "B (mT)",
                "Distance",
                "Distance ratio",
                "SAXS data"
            ])
            self.display_results(self.df)

            QMessageBox.information(self, "Success", "Determination of correlation distances done. Inspect results.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"{e}")

    def plot_radial_profile(self, processor: SAXSDataProcessor):
        self.canvas.figure.clf()
        ax = self.canvas.figure.add_subplot(111)
        nb_peaks = int(self.nb_peaks_input.text())
        azimuth = float(self.azimuth_input.text())
        width = float(self.width_input.text())
        q, I = processor.pyFAI_extract_radialprofiles(azimuth=azimuth, width=width, caving=True)
        q_detected = processor.detect_all_peaks_by_second_derivative(q, I, nb_peaks=nb_peaks)
        
        ax.loglog(q, I, label="I(q)")
        
        # Génération automatique de couleurs distinctes
        colors = plt.cm.tab10(np.linspace(0, 1, len(q_detected)))  # ou tab20, Set1, etc.
        
        for i, qp in enumerate(q_detected):
            ax.axvline(qp, color=colors[i], ls='--', label=f'q={qp:.3f}$\AA^{{-1}}$, d = {0.2*np.pi/qp:.1f}nm')
        
        ax.set_xlabel("q (Å⁻¹)")
        ax.set_ylabel("Intensity")
        ax.grid()
        ax.legend()
        ax.set_title(processor.samplename)
        
        # Mettre à jour la toolbar après le plot
        self.canvas.figure.tight_layout()
        self.toolbar.update()
        self.canvas.draw()
    
    def run_processing(self):
        try:
            if not self.file_list:
                QMessageBox.warning(self, "Error", "No data file selected")
                return

            path = os.path.dirname(self.file_list[0])
            #q1= float(self.q1_input.text())
            #q2= float(self.q2_input.text())
            nb_peaks=int(self.nb_peaks_input.text())                                                                                
            azimuth=float(self.azimuth_input.text())                                                                                
            width=float(self.width_input.text())  # Cette variable manquait                                                         
            qvalues = np.array(list(map(float, self.qvalues_input.text().split(','))))  # Cette variable manquait aussi
            self.progress.setMaximum(len(self.file_list))                                                                           
            self.progress.setValue(0)                                                                                               
            results = []

            for idx, file in enumerate(self.file_list):
                try:
                    processor = self.init_SAXSProcessor(file)
                    sample = processor.samplename
                    B = processor.B
                    data_array = processor.data  # image SAXS
                    # distances de corrélation
                    d= processor.compute_correlation_distance(azimuth=azimuth,width=width,nb_peaks=nb_peaks,caving=True)
                    #slope=processor.slope_determination(q1,q2)

                    #calcule S  

                    result_dict = processor.compute_S()
                    
                    for i, q in enumerate(qvalues):
                        val = result_dict[str(q)]
                        background=val[0]
                        I=val[1]
                        position= val[2]
                        x0_S=val[3]
                        """
                        if position<0 and position <-20:
                            position+=180 # position must be between 0 and 180 (we only consider this range for S calculation)
                        if position >20:
                            position -=90 # center position around 0 so that 0<S<1
                        """
                        gamma =val[4]
                        eta=val[5]
                        slope = val[6]
                        d_val=d[i]/10.0
                        results.append([
                            os.path.basename(file),
                            sample,
                            B,
                            q,
                            d_val,       # correlation distance
                            background,     #y0
                            slope,          # slope for linear background
                            I,              # PV intensity
                            x0_S,           # peak position
                            gamma,          # parameter correlated to peak width
                            eta,            # mixing parameter
                            val[7],         # S
                            val[8],         # R² (S)

                            data_array      # image SAXS (tableau numpy)
                        ])
                    
                    self.plot_azimuthal_profile(processor)
                                        
                    qvalues =np.array([2*np.pi/ dist for dist in d])
                    line2write=','.join(f"{item:.4f}" for item in qvalues)
                    self.qvalues_input.setText(line2write)
                    qvalues = np.array(list(map(float, self.qvalues_input.text().split(','))))
                    
                except Exception as e:
                    print(f"Erreur fichier {file} : {e}")
                self.progress.setValue(idx + 1)

            self.df = pd.DataFrame(results, columns=[
                "File Name",
                "Sample",
                "Champ B (mT)",
                "q (A-1)",
                "Distance (nm)",
                "background y0",
                "background slope",
                "PV intensity",
                "PV position",
                "PV sigma",
                "PV eta (mix)",
                "S",
                "R²",
                "SAXS data"
            ])
            self.display_results(self.df)

            QMessageBox.information(self, "Success", "Determination of S and d done. Inspect results")

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"{e}")

    def run_processing_slope(self):
        try:
            if not self.file_list:
                QMessageBox.warning(self, "Error", "No data file selected")
                return
            q1= float(self.q1_input.text())
            q2= float(self.q2_input.text())
            self.progress.setMaximum(len(self.file_list))
            self.progress.setValue(0)
            results = []

            for idx, file in enumerate(self.file_list):
                try:
                    processor = self.init_SAXSProcessor(file)
                    sample = processor.samplename  # Ces variables manquaient
                    B = processor.B
                    slope=processor.slope_determination(q1,q2)
                    results.append([
                        os.path.basename(file),
                        sample,
                        B,                        
                        slope,            # slope
                        #data_array     # image SAXS (tableau numpy)
                    ])

                except Exception as e:
                    print(f"Erreur fichier {file} : {e}")
                self.progress.setValue(idx + 1)
                
            self.df = pd.DataFrame(results, columns=[
                "File Name",
                "Sample",
                "Champ B (mT)",
                "slope"  # Supprimé "q" car il n'était pas dans les données
            ])
            self.display_results(self.df)

            QMessageBox.information(self, "Success", "Determination of slope done. Inspect results")

        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"{e}")

    def save_results(self):
        if self.df is not None:
            path, _ = QFileDialog.getSaveFileName(self, "Save CSV file", "", "CSV File (*.csv)")
            if path:
                self.df.to_csv(path, index=False)
                QMessageBox.information(self, "Success", f"File saved in : {path}")
        else:
            QMessageBox.warning(self, "Error", "No data to save.")

    def save2png(self):
        try:
            if not self.file_list:
                QMessageBox.warning(self, "Error", "No data file selected")
                return
            
            self.progress.setMaximum(len(self.file_list))
            self.progress.setValue(0)
            
            for idx, file in enumerate(self.file_list):
                try:
                    processor = self.init_SAXSProcessor(file)
                    processor.save_data_to_png()
                except Exception as e:
                    print(f"Erreur fichier {file} : {e}")
                
                self.progress.setValue(idx + 1)
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"{e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = SAXSGUI()
    gui.show()
    sys.exit(app.exec_())