import sys
from os import listdir
import os.path as osp
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
import cv2
import matplotlib.pyplot as plt
import shutil
import os

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util


DEFAULT_IMAGE_ALBUM_DIRECTORY = 'data/data20200402/'
DEFAULT_IMAGE_FACE_DIRECTORY = 'data/data20200402_face/'


# Check that a file name has a valid image extension for QPixmap
def filename_has_image_extension(filename):
    valid_img_extensions = \
        ['bmp', 'gif', 'jpg', 'jpeg', 'png', 'pbm', 'pgm', 'ppm', 'xbm', 'xpm']
    filename = filename.lower()
    extension = filename[-3:]
    four_char = filename[-4:] # exclusively for jpeg
    if extension in valid_img_extensions or four_char in valid_img_extensions:
        return True
    else:
        return False


# Widget for the single image that is currently on display
class DisplayImage(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self)
        self.parent = parent
        self.pixmap = QPixmap()
        self.label = QLabel(self)
        self.assigned_img_full_path = ''

    def update_display_image(self, path_to_image=''):
        self.assigned_img_full_path = path_to_image

        # render the display image when a thumbnail is selected
        self.on_main_window_resize()

    def on_main_window_resize(self, event=None):
        main_window_size = self.parent.size()
        main_window_height = main_window_size.height()
        main_window_width = main_window_size.width()

        display_image_max_height = main_window_height - 50
        display_image_max_width = main_window_width - 400

        self.pixmap = QPixmap(self.assigned_img_full_path)
        self.pixmap = self.pixmap.scaled(\
            QSize(display_image_max_width, display_image_max_height), \
            Qt.KeepAspectRatio, \
            Qt.SmoothTransformation)

        self.label.setPixmap(self.pixmap)


# Widget for selecting an image in the directory to display
# Makes a vertical scrollable widget with selectable image thumbnails
class ImageFileSelector(QWidget):
    def __init__(self, parent=None, album_path='', display_image=None):
        QWidget.__init__(self, parent=parent)
        self.display_image = display_image
        self.grid_layout = QGridLayout(self)
        self.grid_layout.setVerticalSpacing(30)

        # Get all the image files in the directory
        if album_path != '':
            files = [f for f in listdir(album_path) if osp.isfile(osp.join(album_path, f))]
            row_in_grid_layout = 0
            first_img_file_path = ''

            # Render a thumbnail in the widget for every image in the directory
            for file_name in files:
                if filename_has_image_extension(file_name) is False:
                    continue
                img_label = QLabel()
                text_label = QLabel()
                img_label.setAlignment(Qt.AlignCenter)
                text_label.setAlignment(Qt.AlignCenter)
                file_path = album_path + file_name
                pixmap = QPixmap(file_path)
                pixmap = pixmap.scaled(\
                    QSize(100, 100), \
                    Qt.KeepAspectRatio, \
                    Qt.SmoothTransformation)
                img_label.setPixmap(pixmap)
                text_label.setText(file_name)
                img_label.mousePressEvent = \
                    lambda e, \
                    index=row_in_grid_layout, \
                    file_path=file_path: \
                        self.on_thumbnail_click(e, index, file_path)
                text_label.mousePressEvent = img_label.mousePressEvent
                thumbnail = QBoxLayout(QBoxLayout.TopToBottom)
                thumbnail.addWidget(img_label)
                thumbnail.addWidget(text_label)
                self.grid_layout.addLayout( \
                    thumbnail, row_in_grid_layout, 0, Qt.AlignCenter)

                if row_in_grid_layout == 0:
                    first_img_file_path = file_path
                row_in_grid_layout += 1

            # Automatically select the first file in the list during init
            self.on_thumbnail_click(None, 0, first_img_file_path)

    def on_thumbnail_click(self, event, index, img_file_path):
        # Deselect all thumbnails in the image selector
        for text_label_index in range(len(self.grid_layout)):
            text_label = self.grid_layout.itemAtPosition(text_label_index, 0)\
                .itemAt(1).widget()
            text_label.setStyleSheet("background-color:none;")

        # Select the single clicked thumbnail
        text_label_of_thumbnail = self.grid_layout.itemAtPosition(index, 0)\
            .itemAt(1).widget()
        text_label_of_thumbnail.setStyleSheet("background-color:blue;")

        # Update the display's image
        self.display_image.update_display_image(img_file_path)

    def update_img_file_path(self, album_path=''):
        # first clear the layout layout
        for i in range(self.grid_layout.count()):
            item = self.grid_layout.itemAt(0)
            if item is not None:
                for i in range(item.count()):
                    item_item = item.itemAt(0)
                    widget = item_item.widget()
                    if widget is not None:
                        item.removeWidget(widget)
                        widget.deleteLater()
            self.grid_layout.removeItem(item)

        # show the new album
        # Get all the image files in the directory
        files = [f for f in listdir(album_path)]
        row_in_grid_layout = 0

        # Render a thumbnail in the widget for every image in the directory
        for file_name in files:
            if not filename_has_image_extension(file_name):
                break
            img_label = QLabel()
            text_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            text_label.setAlignment(Qt.AlignCenter)
            file_path = osp.join(album_path, file_name)
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(\
                QSize(100, 100), \
                Qt.KeepAspectRatio, \
                Qt.SmoothTransformation)
            img_label.setPixmap(pixmap)
            text_label.setText(file_name)
            img_label.mousePressEvent = \
                lambda e, \
                index=row_in_grid_layout, \
                file_path=file_path: \
                    self.on_thumbnail_click(e, index, file_path)
            text_label.mousePressEvent = img_label.mousePressEvent
            thumbnail = QBoxLayout(QBoxLayout.TopToBottom)
            thumbnail.addWidget(img_label)
            thumbnail.addWidget(text_label)
            self.grid_layout.addLayout( \
                thumbnail, row_in_grid_layout, 0, Qt.AlignCenter)

            if row_in_grid_layout == 0:
                first_img_file_path = file_path
            row_in_grid_layout += 1

        # Automatically select the first file in the list during init
        self.on_thumbnail_click(None, 0, first_img_file_path)


# Widget for selecting an face in the directory to display the list of images
# Makes a vertical scrollable widget with selectable image thumbnails
class ImageWinSelector(QWidget):
    def __init__(self, parent=None, album_path='', display_win=None):
        QWidget.__init__(self, parent=parent)
        # display win have image_file_selector, image_face_selector
        self.display_win = display_win
        self.grid_layout = QGridLayout(self)
        self.grid_layout.setVerticalSpacing(30)

        # Get all the image files in the directory
        files = [f for f in listdir(album_path) ]
        row_in_grid_layout = 0

        # Render a thumbnail in the widget for every image in the directory
        for file_name in files:
            face_display = None
            face_cnt = len(os.listdir(osp.join(album_path, file_name))) - 2
            for face_name in listdir(osp.join(album_path, file_name)):
                if filename_has_image_extension(face_name):
                    face_display = face_name
                    break
            img_label = QLabel()
            text_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            text_label.setAlignment(Qt.AlignCenter)
            file_path = osp.join(album_path, file_name, face_display)
            pixmap = QPixmap(file_path)
            pixmap = pixmap.scaled(\
                QSize(100, 100), \
                Qt.KeepAspectRatio, \
                Qt.SmoothTransformation)
            img_label.setPixmap(pixmap)
            # show the face id and cnt num
            text_label.setText("id: {}, cnt: {:0>2d}".format(file_name, face_cnt))
            img_label.mousePressEvent = \
                lambda e, \
                index=row_in_grid_layout, \
                file_path=file_path: \
                    self.on_thumbnail_click(e, index, file_path)
            text_label.mousePressEvent = img_label.mousePressEvent
            thumbnail = QBoxLayout(QBoxLayout.TopToBottom)
            thumbnail.addWidget(img_label)
            thumbnail.addWidget(text_label)
            self.grid_layout.addLayout( \
                thumbnail, row_in_grid_layout, 0, Qt.AlignCenter)

            if row_in_grid_layout == 0: first_img_file_path = file_path
            row_in_grid_layout += 1

    def on_thumbnail_click(self, event, index, img_file_path):
        # Deselect all thumbnails in the image selector
        for text_label_index in range(len(self.grid_layout)):
            text_label = self.grid_layout.itemAtPosition(text_label_index, 0)\
                .itemAt(1).widget()
            text_label.setStyleSheet("background-color:none;")

        # Select the single clicked thumbnail
        text_label_of_thumbnail = self.grid_layout.itemAtPosition(index, 0)\
            .itemAt(1).widget()
        text_label_of_thumbnail.setStyleSheet("background-color:blue;")

        # update self.display_win.image_face_selector
        album_path = img_file_path.replace('face', 'orig')
        album_path = osp.split(album_path)[0]
        self.display_win.setCurrentIndex(1)
        self.display_win.image_face_selector.update_img_file_path(album_path)


class LeftWin(QTabWidget):
    def __init__(self, parent=None, album_path='', display_image=None):
        super(LeftWin, self).__init__(parent)
        self.setWindowTitle('照片视图')
        self.image_file_selector = ImageFileSelector(album_path=album_path, display_image=display_image)
        self.image_face_selector = ImageFileSelector(album_path='', display_image=display_image)
        # use for display original images
        scroll0 = QScrollArea()
        scroll0.setWidgetResizable(True)
        # scroll.setFixedWidth(140)
        nav0 = scroll0
        nav0.setWidget(self.image_file_selector)

        # use for display one person images
        scroll1 = QScrollArea()
        scroll1.setWidgetResizable(True)
        nav1 = scroll1
        nav1.setWidget(self.image_face_selector)

        self.addTab(nav0, '文件夹')
        self.addTab(nav1, '人像')

        # display the tab pos in north
        self.setTabPosition(QTabWidget.South)


class RightWin(QWidget):
    def __init__(self, parent=None, face_path='', display_scroll=None):
        super(RightWin, self).__init__(parent)
        # need to change to face
        self.face_selector = ImageWinSelector(album_path=face_path, display_win=display_scroll)
        self.image_info = QLabel(self)
        self.image_info.setFixedHeight(200)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        nav = scroll
        nav.setWidget(self.face_selector)

        hbox = QVBoxLayout()
        hbox.addWidget(nav)
        hbox.addWidget(self.image_info)

        self.setLayout(hbox)


class TopMenu(QMenuBar):
    def __init__(self, parent=None, album_path='', display_image=None):
        super(TopMenu, self).__init__(parent)
        self.album_path = album_path
        self.display_image = display_image
        self.opt = TestOptions().parse()  # get test options
        # hard-code some parameters for test
        # test code only supports num_threads = 1
        self.opt.num_threads = 0
        # test code only supports batch_size = 1
        self.opt.batch_size = 1
        # disable data shuffling; comment this line if results on randomly chosen images are needed.
        self.opt.serial_batches = True
        # no flip; comment this line if results on flipped images are needed.
        self.opt.no_flip = True
        # no visdom display; the test code saves the results to a HTML file.
        self.opt.display_id = -1
        # match the generator architecture of the trained model
        self.opt.no_dropout = True
        self.opt.dataroot = "data/transfer_tmp"

        self.opt.name = "style_monet_pretrained"
        # create a model given opt.model and other options
        self.monet_model = create_model(self.opt)
        # regular setup: load and print networks; create schedulers
        self.monet_model.setup(self.opt)

        self.opt.name = "style_vangogh_pretrained"
        self.vango_model = create_model(self.opt)
        self.vango_model.setup(self.opt)

        self.opt.name = "style_cezanne_pretrained"
        self.cezanne_model = create_model(self.opt)
        self.cezanne_model.setup(self.opt)

        self.opt.name = "style_ukiyoe_pretrained"
        self.ukiyoe_model = create_model(self.opt)
        self.ukiyoe_model.setup(self.opt)

        actionTransfer = self.addMenu("Style Transfer")
        monet = QAction("&Monet", self)
        monet.setStatusTip("transform image to monet style")
        monet.triggered.connect(self.monet_transform)
        actionTransfer.addAction(monet)

        VanGogh = QAction("&Van Gogh", self)
        VanGogh.setStatusTip("transform image to Van Gogh style")
        VanGogh.triggered.connect(self.vangogh_transform)
        actionTransfer.addAction(VanGogh)

        Cezanne = QAction("&Cezanne", self)
        Cezanne.setStatusTip("transform image to Cezanne style")
        Cezanne.triggered.connect(self.cezanne_transform)
        actionTransfer.addAction(Cezanne)

        Ukiyo = QAction("&Ukiyo-e", self)
        Ukiyo.setStatusTip("transform image to Ukiyo-e style")
        Ukiyo.triggered.connect(self.ukiyo_transform)
        actionTransfer.addAction(Ukiyo)

    def monet_transform(self, q):
        self.transform_show(self.monet_model)

    def vangogh_transform(self, q):
        self.transform_show(self.vango_model)

    def cezanne_transform(self, q):
        self.transform_show(self.cezanne_model)

    def ukiyo_transform(self, q):
        self.transform_show(self.ukiyoe_model)

    def transform_show(self, model):
        for file in os.listdir(self.opt.dataroot):
            os.remove(osp.join(self.opt.dataroot, file))
        shutil.copy(self.display_image.assigned_img_full_path, self.opt.dataroot)
        orig_image = cv2.imread(self.display_image.assigned_img_full_path)
        height, width, _ = orig_image.shape
        dataset = create_dataset(self.opt)
        model.eval()
        for i, data in enumerate(dataset):
            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            visuals = model.get_current_visuals()  # get image results
            for label, image_data in visuals.items():
                if label == "real":
                    continue
                image = util.tensor2im(image_data)
                image = cv2.resize(image, (width, height))
                plt.imshow(image)
                plt.show()


class App(QWidget):
    def __init__(self):
        super().__init__()
        # Set main window attributes
        self.title = 'Vlog Tool'
        self.left = 0
        self.top = 0
        self.width = 1080
        self.height = 720
        self.resizeEvent = lambda e: self.on_main_window_resize(e)

        # Make 2 widgets, one to select an image and one to display an image
        self.display_image = DisplayImage(self)
        self.left_win = LeftWin(album_path=DEFAULT_IMAGE_ALBUM_DIRECTORY, display_image=self.display_image)
        self.right_win = RightWin(face_path=DEFAULT_IMAGE_FACE_DIRECTORY, display_scroll=self.left_win)

        layout = QGridLayout(self)
        # set the menu
        self.menubar = TopMenu(album_path=DEFAULT_IMAGE_ALBUM_DIRECTORY, display_image=self.display_image)
        layout.addWidget(self.menubar, 0, 0)

        # set the button for style transform
        btn_layout = QGridLayout()
        self.btn_monet = QPushButton("monet")
        self.btn_monet.clicked.connect(self.menubar.monet_transform)
        btn_layout.addWidget(self.btn_monet, 0, 1)
        self.btn_vangogh = QPushButton("vangogh")
        self.btn_vangogh.clicked.connect(self.menubar.vangogh_transform)
        btn_layout.addWidget(self.btn_vangogh, 0, 2)
        self.btn_cezanne = QPushButton("cezanne")
        self.btn_cezanne.clicked.connect(self.menubar.cezanne_transform)
        btn_layout.addWidget(self.btn_cezanne, 0, 3)
        self.btn_ukiyo = QPushButton("ukiyo")
        self.btn_ukiyo.clicked.connect(self.menubar.ukiyo_transform)
        btn_layout.addWidget(self.btn_ukiyo, 0, 4)
        layout.addLayout(btn_layout, 0, 1)

        # Add the 2 widgets to the main window layout
        layout.addWidget(self.left_win, 1, 0, Qt.AlignLeft)
        layout.addWidget(self.display_image.label, 1, 1, Qt.AlignRight)
        layout.addWidget(self.right_win, 1, 2, Qt.AlignRight)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()

    # Set the display image size based on the new window size
    def on_main_window_resize(self, event):
        self.display_image.on_main_window_resize(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
