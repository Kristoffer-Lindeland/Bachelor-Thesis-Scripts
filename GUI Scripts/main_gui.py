import sys
import os
import time
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QHBoxLayout, QComboBox, QTextEdit, QCheckBox, QMessageBox, QFrame,
    QSizePolicy, QProgressBar, QGroupBox
)
from PyQt5.QtCore import QPoint
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

if getattr(sys, 'frozen', False):
    # Running as exe
    BASE_DIR = sys._MEIPASS
else:
    # Running as python script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
ALGORITHMS = ["nsf5", "uerd", "j_uniward"]
ML_MODELS = ["Random Forest", "SVM", "CNN", "MLP"]
FIXED_RATE = "04"

print("Loading model from:", MODEL_DIR)


class SyncedTextEdit(QTextEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.synced_boxes = []

    # used for syncing the scroll bars of the three result boxes
    def wheelEvent(self, event):
        super().wheelEvent(event)
        value = self.verticalScrollBar().value()
        for box in self.synced_boxes:
            box.verticalScrollBar().setValue(value)

    # used for syncing line selection when clicking on any of the result boxes
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        cursor = self.cursorForPosition(event.pos())
        line_number = cursor.blockNumber()
        self.select_line(line_number)
        for box in self.synced_boxes:
            box.select_line(line_number)

    # used for syncing line selection when using arrow keys in the result boxes
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Up, Qt.Key_Down):
            cursor = self.textCursor()
            line_number = cursor.blockNumber()
            if event.key() == Qt.Key_Up:
                line_number = max(0, line_number - 1)
            elif event.key() == Qt.Key_Down:
                line_number += 1
            self.select_line(line_number)
            for box in self.synced_boxes:
                box.select_line(line_number)
        else:
            super().keyPressEvent(event)

    # highlights a specific line in the text box
    def select_line(self, line_number):
        cursor = self.textCursor()
        cursor.movePosition(cursor.Start)
        for _ in range(line_number):
            cursor.movePosition(cursor.Down)
        cursor.select(cursor.LineUnderCursor)
        self.setTextCursor(cursor)


# main GUI class, sets up the window and initializes everything
class StegoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Steganalyser 1.0")
        self.setGeometry(100, 100, 870, 600)

        self.results = []
        self.current_files = []


        self.stego_count = 0
        self.cover_count = 0

        self.init_ui()
        self.setAcceptDrops(True)

    # sets up all the UI elements and layout
    def init_ui(self):
        layout = QHBoxLayout()
        controls = QVBoxLayout()

        # dropdown to select which machine learning model to use
        model_group = QGroupBox("Machine Learning Model")
        model_layout = QVBoxLayout()
        self.ml_model_dropdown = QComboBox()
        self.ml_model_dropdown.addItems(ML_MODELS)
        self.ml_model_dropdown.setCurrentText("Random Forest")
        self.ml_model_dropdown.currentTextChanged.connect(self.update_model_status)
        model_layout.addLayout(self.with_help(self.ml_model_dropdown, "Select the machine learning model to use."
                                                                      "\n\nNote: \nCurrently only the Random Forest model\n"
                                                                      "is available."))

        # adds the model selection dropdown and help button to the tool
        model_group.setLayout(model_layout)
        controls.addWidget(model_group)

        # dropdown to select which steganography algorithm to test against
        alg_group = QGroupBox("Select target algorithm")
        alg_layout = QVBoxLayout()
        self.alg_dropdown = QComboBox()
        self.alg_dropdown.addItems(ALGORITHMS + ["All"])
        self.alg_dropdown.setCurrentText("All")
        self.alg_dropdown.currentTextChanged.connect(self.update_model_status)
        alg_layout.addLayout(self.with_help(self.alg_dropdown, "Select steganography algorithm to test against."
                                                               "\n\nTip: \nFor best results choose 'All'"))

        # adds the algorithm selection dropdown and help button to the tool
        alg_group.setLayout(alg_layout)
        controls.addWidget(alg_group)

        # buttons for selecting image or folder
        file_group = QGroupBox("Select images (only JPEG)")
        file_layout = QVBoxLayout()
        self.select_button = QPushButton("Select Single Image")
        self.select_button.clicked.connect(lambda: self.select_path(file_mode=True))
        file_layout.addLayout(self.with_help(self.select_button, "Select individual JPEG image(s) to analyze."))
        self.select_folder_button = QPushButton("Select Folder")
        self.select_folder_button.clicked.connect(lambda: self.select_path(file_mode=False))
        file_layout.addLayout(self.with_help(self.select_folder_button, "Select a folder containing JPEG images that will be analyzed."))

        # drag & drop box
        self.drop_hint = QLabel("Drag & Drop Images ")
        self.drop_hint.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                padding: 20px;
                margin-top: 5px;
                color: #666;
                font-style: italic;
                background-color: #f9f9f9;
            }
        """)

        # centers the drag & drop text
        self.drop_hint.setAlignment(Qt.AlignCenter)
        file_layout.addWidget(self.drop_hint)

        # adds the buttons for file selection and drag & drop to the tool
        file_group.setLayout(file_layout)
        controls.addWidget(file_group)

        # checkboxes to filter results and a button to export results as CSV
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        self.filter_stego = QCheckBox("Show Stego")
        self.filter_stego.setChecked(True)
        self.filter_cover = QCheckBox("Show Cover")
        self.filter_cover.setChecked(True)
        self.filter_stego.stateChanged.connect(self.filter_results)
        self.filter_cover.stateChanged.connect(self.filter_results)
        options_layout.addWidget(self.filter_stego)
        options_layout.addWidget(self.filter_cover)
        self.save_button = QPushButton("Save Results as CSV")
        self.save_button.clicked.connect(self.save_csv)
        options_layout.addLayout(self.with_help(self.save_button, "Export the results to a CSV file."))
        options_group.setLayout(options_layout)
        controls.addWidget(options_group)

        # progress bar
        self.progress_bar = QProgressBar()
        controls.addWidget(self.progress_bar)

        # label to display estimated time remaining
        self.time_label = QLabel("Estimated time remaining: --")
        controls.addWidget(self.time_label)

        # layout for analysis info section and play button
        info_play_layout = QVBoxLayout()
        info_play_layout.addSpacing(10)

        # title for the analysis info section
        info_title = QLabel("Analysis Information")
        font = QFont()
        font.setBold(True)
        info_title.setFont(font)
        info_play_layout.addWidget(info_title)
        info_row = QHBoxLayout()

        # label that shows current model, algorithm, and image stats during analysis
        self.info_display = QLabel()
        self.info_display.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.info_display.setMinimumHeight(100)
        self.info_display.setFont(QFont("Arial", 10))
        self.info_display.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        info_row.addWidget(self.info_display, 3)

        # centered button to start the analysis
        self.play_button = QPushButton("▶\nStart Analysis")
        play_font = QFont("Arial", 10)
        play_font.setBold(True)
        self.play_button.setFont(play_font)
        self.play_button.setStyleSheet("text-align: center;")
        self.play_button.setFixedHeight(self.info_display.minimumHeight())
        self.play_button.clicked.connect(self.run_analysis)
        info_row.addWidget(self.play_button, 1)

        # add the info box and play button
        info_play_layout.addLayout(info_row)
        controls.addLayout(info_play_layout)

        # completes the left control panel and adds it to the tool
        controls.addStretch()
        controls_container = QWidget()
        controls_container.setLayout(controls)
        controls_container.setFixedWidth(500)
        layout.addWidget(controls_container, alignment=Qt.AlignLeft)

        # layout for the right side of the window
        right_panel = QVBoxLayout()

        # labels for the three result columns
        self.filename_label = QLabel("<b>Filename</b>")
        self.prediction_label = QLabel("<b>Prediction</b>")
        self.confidence_label = QLabel("<b>Confidence</b>")

        # text boxes for showing filenames, predictions, and confidence scores
        self.filename_box = SyncedTextEdit()
        self.prediction_box = SyncedTextEdit()
        self.confidence_box = SyncedTextEdit()

        # link the three result boxes together so scrolling and selection stay in sync
        self.filename_box.synced_boxes = [self.prediction_box, self.confidence_box]
        self.prediction_box.synced_boxes = [self.filename_box, self.confidence_box]
        self.confidence_box.synced_boxes = [self.filename_box, self.prediction_box]

        # style for highlighting selected lines
        highlight_style = """
        QTextEdit {
            selection-background-color: #cceeff;
            selection-color: black;
        }
        """

        # apply the highlight style
        self.filename_box.setStyleSheet(highlight_style)
        self.prediction_box.setStyleSheet(highlight_style)
        self.confidence_box.setStyleSheet(highlight_style)

        # configure filename box
        self.filename_box.setReadOnly(True)
        self.filename_box.setLineWrapMode(QTextEdit.NoWrap)
        self.filename_box.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.filename_box.setFixedWidth(150)

        # configure prediction box
        self.prediction_box.setReadOnly(True)
        self.prediction_box.setLineWrapMode(QTextEdit.NoWrap)
        self.prediction_box.setFixedWidth(100)

        # configure confidence box
        self.confidence_box.setReadOnly(True)
        self.confidence_box.setLineWrapMode(QTextEdit.NoWrap)
        self.confidence_box.setFixedWidth(100)

        # hide scrollbars from filename and prediction boxes
        self.filename_box.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.prediction_box.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # when scrolling the confidence box, sync the scroll position of other boxes
        self.confidence_box.verticalScrollBar().valueChanged.connect(
            lambda value: self.filename_box.verticalScrollBar().setValue(value)
        )
        self.confidence_box.verticalScrollBar().valueChanged.connect(
            lambda value: self.prediction_box.verticalScrollBar().setValue(value)
        )

        # layout for the three result boxes
        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(0)

        # layout for the filename box
        column1 = QVBoxLayout()
        column1.addWidget(self.filename_label)
        column1.addWidget(self.filename_box)

        # layout for the prediction box
        column2 = QVBoxLayout()
        column2.addWidget(self.prediction_label)
        column2.addWidget(self.prediction_box)

        # layout for the confidence box
        column3 = QVBoxLayout()
        column3.addWidget(self.confidence_label)
        column3.addWidget(self.confidence_box)

        # add all three result boxes to the tool
        columns_layout.addLayout(column1)
        columns_layout.addLayout(column2)
        columns_layout.addLayout(column3)

        # add the result boxes to the right panel and the main layout of the window
        right_panel.addLayout(columns_layout)
        layout.addLayout(right_panel)
        self.setLayout(layout)

        # update the model status display based on current selections and loaded files
        self.update_model_status()

    def with_help(self, widget, help_text):
        # creates a  help button with a ? in it
        layout = QHBoxLayout()
        layout.addWidget(widget)
        help_button = QPushButton("?")
        help_button.setFixedSize(20, 20)

        # show a pop up with help text when help button is clicked
        def show_context_help():
            msg = QMessageBox(self)
            msg.setWindowTitle("Help")
            msg.setText(help_text)
            msg.setIcon(QMessageBox.Information)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.adjustSize()

            # position the help pop-up close to the help button and show it
            global_pos = help_button.mapToGlobal(QPoint(0, help_button.height()))
            msg.move(global_pos + QPoint(10, 10))
            msg.exec_()

        # connect the help button to show the pop-up and add it to the tool
        help_button.clicked.connect(show_context_help)
        layout.addWidget(help_button)
        return layout

    # updates the info box with current model, algorithm, and image stats
    def update_model_status(self, stego_count=None, cover_count=None):
        model_selected = self.ml_model_dropdown.currentText()
        alg_selected = self.alg_dropdown.currentText()

        # if another model than Random Forest is selected, show a warning and disable the start button
        if model_selected != "Random Forest":
            self.info_display.setText(
                "<div style='text-align:center;'>"
                "<font color='red'><b>-- WARNING --</b><br>"
                "Model not imported<br>Select Random Forest</font>"
                "</div>"
            )
            self.play_button.setEnabled(False)

        else:
            # if Random Forest is selected, prepare updated stats for images, stego, and cover counts
            image_info = "<font color='red'>0</font>" if not self.current_files else str(len(self.current_files))
            stego_text = str(self.stego_count)
            cover_text = str(self.cover_count)

            # formating display current model, algorithm, and image stats
            info = (
                "<pre>"
                f"<b>Using model:</b>      {model_selected}<br>"
                f"<b>Target algorithm:</b> {alg_selected}<br>"
                f"<b>Images loaded:</b>    {image_info}<br>"
                f"<b>Stego images:</b>     {stego_text}<br>"
                f"<b>Cover images:</b>     {cover_text}<br>"
                "</pre>"
            )
            # update the info display and enable the play button if images are loaded
            self.info_display.setText(info)
            self.play_button.setEnabled(bool(self.current_files))

    def select_path(self, file_mode=True):
        # if selecting single files, open file browser and update the list of current files
        if file_mode:
            files, _ = QFileDialog.getOpenFileNames(self, "Select JPEG Image", filter="Images (*.jpg *.jpeg)")
            if files:
                self.current_files = files
                self.update_model_status()
        else:
            # same but for folder
            folder = QFileDialog.getExistingDirectory(self, "Select Folder")
            if folder:
                self.current_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg"))]
                self.update_model_status()

    # runs the analysis on the selected images
    def run_analysis(self):
        try:
            from model_rf import predict_rf
            # clear previous results and reset progress bar and timer display
            self.filename_box.clear()
            self.prediction_box.clear()
            self.confidence_box.clear()
            self.results = []
            self.progress_bar.setValue(0)
            self.time_label.setText("Estimated time remaining: --")

            # Reset counters
            self.stego_count = 0
            self.cover_count = 0
            self.update_model_status()

            # record the start time
            start_time = time.time()

            # get selected algorithm and model from the dropdown
            alg_selected = self.alg_dropdown.currentText()
            model_selected = self.ml_model_dropdown.currentText()
            algs = ALGORITHMS if alg_selected == "All" else [alg_selected]

            # get total number of images to process completion counter
            total = len(self.current_files)
            completed = 0

            # loop through all selected files and run predictions
            for filepath in self.current_files:
                if model_selected == "Random Forest":
                    predictions = predict_rf(filepath, algs, FIXED_RATE, MODEL_DIR)
                else:
                    predictions = []

                # add the predictions for the image to the results
                self.results.extend(predictions)

                # update stego and cover counters
                for _, label, _, _ in predictions:
                    if label == "Stego":
                        self.stego_count += 1
                    elif label == "Cover":
                        self.cover_count += 1

                # refresh the info display with updated image and label counts
                self.update_model_status()

                # increase the completed counter and calculate elapsed time
                completed += 1
                elapsed = time.time() - start_time

                # estimate and display remaining time
                if completed < total:
                    est_total = elapsed / completed * total
                    est_remaining = est_total - elapsed
                    self.time_label.setText(f"Estimated time remaining: {int(est_remaining)}s")

                else:
                    # if completed, set remaining time to 0
                    self.time_label.setText("Estimated time remaining: 0s")

                # update progress bar and refresh during processing
                self.progress_bar.setValue(int(completed / total * 100))
                QApplication.processEvents()

            # when done, show the total time taken
            total_duration = time.time() - start_time
            self.time_label.setText(f"Total time: {int(total_duration)}s")

            # apply the current filter results
            self.filter_results()

            # recount stego and cover images from the final results and update the display
            self.stego_count = sum(1 for _, label, _, _ in self.results if label == "Stego")
            self.cover_count = sum(1 for _, label, _, _ in self.results if label == "Cover")
            self.update_model_status()

        except Exception as e:
            # show a pop up error message if something goes wrong during analysis
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")

    # adjusts the width of the filename box based on its content
    def adjust_result_box_widths(self):
        font_metrics = self.filename_box.fontMetrics()

        # calculate the needed width based on longest line
        def get_max_width(lines, label):
            text_width = max((font_metrics.horizontalAdvance(line) for line in lines), default=100)
            label_width = font_metrics.horizontalAdvance(label.text())
            return max(text_width, label_width) + 20

        # set the minimum width of the filename box
        filename_lines = self.filename_box.toPlainText().splitlines()
        self.filename_box.setMinimumWidth(get_max_width(filename_lines, self.filename_label))

    # clears and fills the result boxes based on the filters
    def filter_results(self):
        self.filename_box.clear()
        self.prediction_box.clear()
        self.confidence_box.clear()

        # check the checkboxes
        show_stego = self.filter_stego.isChecked()
        show_cover = self.filter_cover.isChecked()

        # loop through results and display if they match the filters
        for filepath, label, conf, _ in self.results:
            if (label == "Stego" and show_stego) or (label == "Cover" and show_cover):
                self.filename_box.append(os.path.basename(filepath))
                self.prediction_box.append(label)
                self.confidence_box.append(f"{conf:.2%}")

        # adjust the widths of the result boxes based on the content
        self.adjust_result_box_widths()

    # if there are any results, save them as CSV
    def save_csv(self):
        if not self.results:
            QMessageBox.warning(self, "Warning", "No results to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", filter="CSV Files (*.csv)")

        # save and show a confirmation message
        if path:
            df = pd.DataFrame(self.results, columns=["Filename", "Prediction", "Confidence", "Model Used"])
            df.to_csv(path, index=False)
            QMessageBox.information(self, "Saved", f"Results saved to {path}")

    # checks if the dragged files contains file paths
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

            # updates the drop hint style to visually indicate that files can be dragged and dropped
            self.drop_hint.setStyleSheet("""
                QLabel {
                    border: 2px dashed #007ACC;
                    padding: 20px;
                    margin-top: 5px;
                    color: #007ACC;
                    font-style: italic;
                    background-color: #e6f2ff;
                }
            """)

    # handles the drop event, getting file paths for dropped images
    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls() if
                 u.toLocalFile().lower().endswith((".jpg", ".jpeg"))]
        # if JPEG files are dropped, update the current files and refresh the model
        if files:
            self.current_files = files
            self.update_model_status()
        self.reset_drop_hint_style()

    # resets the drop hint style when the dragged files leaves the window
    def dragLeaveEvent(self, event):
        self.reset_drop_hint_style()

    # resets the drop hint style to its default state
    def reset_drop_hint_style(self):
        self.drop_hint.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                padding: 20px;
                margin-top: 5px;
                color: #666;
                font-style: italic;
                background-color: #f9f9f9;
            }
        """)

# create the main window, and start the tool
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = StegoApp()
    win.show()
    sys.exit(app.exec_())
