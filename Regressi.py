from math import *
import numpy as np
import matplotlib.pyplot as plt
from random import gauss

def Regressi(L, M, DL, DM, format='y=ax+b', iterations=1000):
    '''
    Fonction pour faire des regressions linéaires avec propagation des incertitudes
    '''

    fig, axs = plt.subplots(2, 2)

    def LookForType(format, incert=False):
        '''
        Fonction qui regarde le type de regression linéaire demandé
        Default:
            y=ax+b
        Exemple:
            y=asqrt(x)+b
        '''

        types = {
            'y=ax+b': lambda X: X,
            'y=asqrt(x)+b': lambda X: [sqrt(_) for _ in X],
            'y=a/x+b': lambda X: [1/_ for _ in X],
            'y=aln(x)+b': lambda X: [log(_) for _ in X]
        }

        incert_types = {
            'y=ax+b': lambda X: X,
            'y=asqrt(x)+b': lambda X: [X[i]*DL[i]/2*L[i] for i in range(len(X))],
            'y=a/x+b': lambda X: [X[i]*DL[i]/L[i] for i in range(len(X))],
            'y=aln(x)+b': lambda X: [log(_) for _ in X]
        }

        if format in types:
            return types.get(format) if incert == False else incert_types.get(format)
        else:
            raise BadRegressionType(f'{format} is not a valid format')

    def ellipse(xc, a, yc, b):
        '''
        Création des ellipses d'incertitudes autour d'un point de coordonnées (xc, yc)
        '''

        te = np.linspace(0, 2*pi, 200)
        xe = [xc + a*cos(_) for _ in te]
        ye = [yc + b*sin(_) for _ in te]
        axs[0][0].plot(xe, ye, 'k', lw=1)

    X, Y = LookForType(format)(L), M
    DX, DY = LookForType(format, True)(DL), DM
    Aalea, Balea = [], []
    x = np.linspace(X[0], X[-1], 100)

    for i in range(len(X)):
        ellipse(X[i], DX[i], Y[i], DY[i])
    axs[0][0].scatter(X, Y)

    for i in range(iterations):
        Xalea = [gauss(X[_], DX[_]) for _ in range(len(X))]
        Yalea = [gauss(Y[_], DY[_]) for _ in range(len(Y))]
        a, b = np.polyfit(Xalea, Yalea, 1)
        Aalea.append(a); Balea.append(b)

        axs[0][1].plot(x, [a*_ + b for _ in x])

    M, m = [], []
    for i in x:
        M.append(max([Aalea[_]*i + Balea[_] for _ in range(len(Aalea))]))
        m.append(min([Aalea[_]*i + Balea[_] for _ in range(len(Aalea))]))

    axs[0][1].plot(x, M, 'k--'); axs[0][1].plot(x, m, 'k--')
    axs[1][0].hist(Aalea); axs[1][0].hist(Balea)

    get_r = lambda X: (np.mean(X),np.std(X))
    moy_a, et_a = get_r(Aalea)
    moy_b, et_b = get_r(Balea)

    axs[0][0].plot(x, [moy_a*_ + moy_b for _ in x], 'r', label=f'y={round(moy_a, 2)}x + {round(moy_b, 2)}')
    axs[1][1].plot(X, DX); axs[1][1].plot(Y, DY)

    '''
    Exceptions
    '''

    class BadRegressionType(Exception):
        pass


    '''
    Legends
    '''
    
    axs[0][0].set_title('Regression')
    axs[0][0].legend()
    axs[0][1].set_title('Enveloppe')
    axs[1][0].set_title('Histogramme a/b')
    axs[1][0].legend(['a', 'b'])
    axs[1][1].set_title('Incertitudes')
    axs[1][1].legend(['DX=f(X)', 'DY=f(Y)'])
    plt.show()

Regressi([1, 2, 3], [1, 2, 3], [0.01, 0.02, 0.01], [0.1, 0.2, 0.1], 'y=ax+b')