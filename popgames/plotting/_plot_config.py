import matplotlib
import matplotlib.pyplot as plt

DPI = 200
FIGSIZE = (3, 3)
FONTSIZE = 10
matplotlib.rcParams['figure.dpi'] = DPI
matplotlib.rcParams['figure.figsize'] = FIGSIZE
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.size': FONTSIZE})