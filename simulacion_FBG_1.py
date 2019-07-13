import time

from numba import jit
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)

c = 3e8
class App(QWidget):

    def __init__(self):
        super().__init__()
        self.left = 10
        self.top = 10
        self.title = 'Simple FBG Simulator'
        self.width = 700
        self.height = 500
        self.Lambda = 1550
        self.L = 3
        self.h = 0.01
        self.ng = 1.45
        self.vg = c/self.ng
        self.tol = 1e-3
        self.f = 100
        self.gauss = 0
        self.k0 = 400
        self.F = 0
        self.canvas = None
        self.initUI()


    def initUI(self):
        grid = QGridLayout()

        dynamic_canvas = FigureCanvas(Figure(figsize=(20, 10)))
        grid.addWidget(dynamic_canvas,0,2, 7, 8)
        self._dynamic_ax = dynamic_canvas.figure.subplots(2,1)
        self._dynamic_ax[1].set_xlabel('f (GHz)')
        self._dynamic_ax[0].set_ylabel('Reflectivity (dB)')
        self._dynamic_ax[1].set_ylabel('Delay (ps)')


        labelLambda = QLabel('lambda (nm)', self)
        labelLambda.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(labelLambda, 0, 0)
        editLambda = QLineEdit(self)
        editLambda.setText(str(self.Lambda))
        grid.addWidget(editLambda, 0, 1)

        labelL = QLabel('L (cm)', self)
        labelL.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(labelL, 1, 0)
        editL = QLineEdit(self)
        editL.setText(str(self.L))
        grid.addWidget(editL, 1, 1)

        labelStep = QLabel('Step (cm)', self)
        labelStep.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(labelStep, 2, 0)
        editStep = QLineEdit(self)
        editStep.setText(str(self.h))
        grid.addWidget(editStep, 2, 1)

        labelng = QLabel('ng', self)
        labelng.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(labelng, 3, 0)
        editng = QLineEdit(self)
        editng.setText(str(self.ng))
        grid.addWidget(editng, 3, 1)

        labelf = QLabel('f (GHz)', self)
        labelf.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(labelf, 4, 0)
        editf = QLineEdit(self)
        editf.setText(str(self.f))
        grid.addWidget(editf, 4, 1)

        labelTol = QLabel('Tol', self)
        labelTol.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        grid.addWidget(labelTol, 5, 0)
        editTol = QLineEdit(self)
        editTol.setText(str(self.tol))
        grid.addWidget(editTol, 5, 1)

        buttonSimular = QPushButton('Run', self)
        buttonSimular.setToolTip('Iniciar simulacion')
        buttonSimular.clicked.connect(lambda: self.simular_button({'editLambda': editLambda.text(),
                                                        'editL': editL.text(),
                                                        'editStep': editStep.text(),
                                                        'editng': editng.text(),
                                                        'editf': editf.text(),
                                                        'editTol': editTol.text()}))
        grid.addWidget(buttonSimular, 6, 0, 1, 2)

        self.setLayout(grid)
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.show()

    def simular_button(self, cadena):
        # print(cadena)
        # print(self.Lambda)
        self.Lambda = float(cadena['editLambda'])*1e-9
        self.L = float(cadena['editL'])*1e-2
        self.h = -float(cadena['editStep'])*1e-2
        self.ng = float(cadena['editng'])
        self.f = float(cadena['editf'])*1e9
        self.Tol = float(cadena['editTol'])
        self.vg = c/self.ng

        # print(self.Lambda)
        # print(c)
        self.simulacion()

    def simulacion(self):
        # print('self.f:' + str(self.f))
        f = np.linspace(-self.f, self.f, 10000)
        # print('f:' + str(f[0]))
        delta = 2*np.pi*f/self.vg
        z = np.arange(0, -(self.L)+self.h, self.h)
        phi_z = 0
        S0 = np.zeros((1,len(f)))
        R0 = np.ones((1,len(f)))

        Sn = S0
        Rn = R0
        # print('S0: '+ str(S0))
        # print('S0: '+ str(S0.shape))
        # print('Sn: '+ str(Sn))
        # print('Sn: '+ str(Sn.shape))
        # print('delta:'+str(delta[0]))

        for i in range(len(z)):
            #print(i)

            dR1 = 1j*delta*Rn + 1j*Sn*self.k0*np.exp(1j*phi_z)
            dS1 = -1j*delta*Sn - 1j*Rn*self.k0*np.exp(-1j*phi_z)

            R1 = Rn - self.h*dR1
            S1 = Sn - self.h*dS1

            dR2 = 1j*delta*R1 + 1j*S1*self.k0*np.exp(1j*phi_z)
            dS2 = -1j*delta*S1 - 1j*R1*self.k0*np.exp(-1j*phi_z)

            R2 = R1 - self.h*dR2
            S2 = S1 - self.h*dS2

            dR3 = 1j*delta*R2 + 1j*S2*self.k0*np.exp(1j*phi_z)
            dS3 = -1j*delta*S2 - 1j*R2*self.k0*np.exp(-1j*phi_z)

            deltaR = self.h*(dR1 + 4*dR3 + dR2)/6
            deltaS = self.h*(dS1 + 4*dS3 + dS2)/6

            errorR = self.h*(dR1 - 2*dR3 + dR2)
            errorS = self.h*(dS1 - 2*dS3 + dS2)

            Rn = Rn - deltaR
            Sn = Sn - deltaS

        coef = np.transpose(Sn/Rn)
        tecta = np.unwrap(np.angle(coef))
        #print(tecta.shape)
        ret = -np.gradient(tecta[:,0], 2*np.pi*f)
        print(ret*1e12)
        #print(tecta[:,0].shape)
        #print(f.shape)
        #plt.plot(f*1e-9, np.transpose(10*np.log10(abs(coef))))
        #plt.show()
        self._update_canvas(f*1e-9, 20*np.log10(abs(coef)), ret*1e12)

    def _update_canvas(self, x, y, y2):
        self._dynamic_ax[0].clear()
        self._dynamic_ax[0].plot(x, y)
        self._dynamic_ax[0].set_ylabel('Reflectivity (dB)')
        self._dynamic_ax[0].grid()
        self._dynamic_ax[0].set_xlim([-self.f*1e-9, self.f*1e-9])
        self._dynamic_ax[0].figure.canvas.draw()

        self._dynamic_ax[1].clear()
        self._dynamic_ax[1].plot(x, y2)
        self._dynamic_ax[1].set_xlabel('f (GHz)')
        self._dynamic_ax[1].set_ylabel('Delay (ps)')
        self._dynamic_ax[1].grid()
        self._dynamic_ax[1].set_xlim([-self.f*1e-9, self.f*1e-9])
        self._dynamic_ax[1].figure.canvas.draw()




if  __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
