__help__ = '''
Simple linear regression package,

@format:

    default:
     - y=ax+b

    all:
     - y=ax+b
     - y=a/x+b
     - y=asqrt(x)+b
     - y=ax**n+b # n is an integer


@params:
    
    - L: list (required)
    - M: list (required)
    - DL: list (required)
    - DM: list (required)
    - format: str (optional)
    - iterations: int (optional)
    - graph: bool (optional)


@output:

    LinearRegResult type:
     - result[key] -> float
     - result.infof(other_result, x) -> bool
     - result.overof(other_result, x) -> bool
'''

'''
Needed libraries, please make sure that you installed all of them
'''
from math import *
import numpy as np
import matplotlib.pyplot as plt
from random import gauss

def regressi(L: list, M: list, DL: list, DM: list, format='y=ax+b', iterations=1000, graph=True):
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
            'y=aln(x)+b': lambda X: [log(_) for _ in X],
        }

        incert_types = {
            'y=ax+b': lambda X: DL,
            'y=asqrt(x)+b': lambda X: [X[i]*DL[i]/2*L[i] for i in range(len(X))],
            'y=a/x+b': lambda X: [X[i]*DL[i]/L[i] for i in range(len(X))],
            'y=aln(x)+b': lambda X: [log(_) for _ in X]
        }

        degrees = {
            '1/2': lambda X: [sqrt(_) for _ in X]
        }

        incert_degrees = {
            '1/2': lambda X: [X[i]*DL[i]/2*L[i] for i in range(len(X))]
        }

        if format in types:
            return types.get(format) if not incert else incert_types.get(format)
        elif '**' in format:
            new_format = format.split('**')
            new_format = new_format[-1].split('+')
            degree = new_format[0]
            try:
                return lambda X: [_**int(degree) for _ in X] if not incert else [int(degree)*X[i]*DL[i]/L[i] for i in range(len(X))]
            except TypeError:
                if degree in degrees:
                    return degrees.get(degree) if not incert else incert_degrees.get(degree)
                else:
                    raise Error.BadRegressionType(f'{format} is not a valid format')
        else:
            raise Error.BadRegressionType(f'{format} is not a valid format')

    def ellipse(xc, a, yc, b):
        '''
        Création des ellipses d'incertitudes autour d'un point de coordonnées (xc, yc)
        '''

        te = np.linspace(0, 2*pi, 200)
        xe = [xc + a*cos(_) for _ in te]
        ye = [yc + b*sin(_) for _ in te]
        axs[0][0].plot(xe, ye, 'k', lw=1)

    X, Y = LookForType(format)(L), M
    DX, DY = LookForType(format, True)(X), DM
    Aalea, Balea = [], []
    x = np.linspace(X[0], X[-1], 100)

    for i in range(len(X)):
        ellipse(X[i], DX[i], Y[i], DY[i])

    for i in range(iterations):
        Xalea = [gauss(X[_], DX[_]) for _ in range(len(X))]
        Yalea = [gauss(Y[_], DY[_]) for _ in range(len(Y))]
        a, b = np.polyfit(Xalea, Yalea, 1)
        Aalea.append(a); Balea.append(b)

        axs[0][1].plot(x, [a*_ + b for _ in x], lw=1)

    M, m = [], []
    for i in x:
        M.append(max([Aalea[_]*i + Balea[_] for _ in range(len(Aalea))]))
        m.append(min([Aalea[_]*i + Balea[_] for _ in range(len(Aalea))]))

    get_r = lambda X: (np.mean(X),np.std(X))
    moy_a, et_a = get_r(Aalea)
    moy_b, et_b = get_r(Balea)


    '''
    Légendes
    '''
    
    if graph:
        axs[0][1].plot(x, M, 'k--'); axs[0][1].plot(x, m, 'k--')
        axs[1][0].hist(Aalea); axs[1][0].hist(Balea)
        axs[0][0].plot(x, [moy_a*_ + moy_b for _ in x], 'r', lw=1)
        axs[1][1].plot(X, DX); axs[1][1].plot(Y, DY)
        axs[0][0].scatter(X, Y, c='k', s=10)

        axs[0][0].set_title('Regression')
        axs[0][0].text(X[0], Y[-1], f'y={round(moy_a, 2)}x+{round(moy_b, 2)}\nΔa={round(et_a,1)}\nΔb={round(et_b,1)}',
            horizontalalignment='left', verticalalignment='top')
        axs[0][1].set_title('Enveloppe')
        axs[1][0].set_title('Histogramme a/b')
        axs[1][0].legend(['a', 'b'])
        axs[1][1].set_title('Incertitudes')
        axs[1][1].legend(['DX=f(X)', 'DY=f(Y)'])
        fig.canvas.manager.set_window_title('Regressi')
        
        plt.show()

    return LinearRegResult(moy_a, moy_b, et_a, et_b)


class LinearRegResult:
    '''
    Objet python qui contient le résultat de la regression linéaire
    '''

    def __init__(self, a, b, da, db):
        self.values = {
            0: a,
            1: b,
            2: da,
            3: db
        }

    def __str__(self):
        return f"a={round(self.values.get(0), 2)}±{round(self.values.get(2), 1)}, b={round(self.values.get(1), 2)}±{round(self.values.get(3), 1)}"

    def __getitem__(self, key: int):
        return self.values.get(key)

    def infof(self, v, x):
        if type(v) == type(LinearRegResult):
            return True if x*(self.values.get(0)-v.values.get(0)) <= v.values.get(1)-self.values.get(1) else False
        else:
            raise BadType(f'{v} is not a valid {type(LinearRegResult)}')

    def overof(self, v, x):
        if type(v) == type(LinearRegResult):
            return True if x*(self.values.get(0)-v.values.get(0)) >= v.values.get(1)-self.values.get(1) else False
        else:
            raise BadType(f'{v} is not a valid {type(LinearRegResult)}')


class Error:
    '''
    Exceptions
    '''

    class BadRegressionType(Exception):
        pass

    class BadType(Exception):
        pass


#r = regressi([50E-3, 70E-3, 100E-3], [4/10, 4.5/10, 5.2/10], [1E-4, 1E-4, 1E-4], [0.03, 0.03, 0.03], 'y=ax+b', 1000, True)
print(__help__)