import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time


def segment(theta, ns):
    """ Segment un vecteur theta
    Cette fonction sert à segmenter un theta en une 
    liste de matrices qui est spécifié par ns qui 
    donne le nombre de neurones par couche

    Parameters
    ----------
    theta : torch.Tensor
        Le concatenation des paramètres du réseaux
    ns : list 
        La liste des nombres de neurones pour chaque
        couche du réseaux

    Returns
    -------
    list
        La liste des tenseurs à chaque couche du réseau
    """

    total_len = np.sum([(ns[i]+1)*ns[i+1] for i in range(len(ns)-1)])
    if (total_len != len(theta)):
        print("Error : Wrong dimensions\n")
        return

    param_list = []
    index = 0
    for i in range(len(ns)-1):

        W = torch.Tensor(
            theta[index: (index + ns[i]*ns[i + 1])]).reshape(ns[i+1], ns[i])
        index += ns[i]*ns[i + 1]

        b = torch.Tensor(theta[index: (index + ns[i+1])]).reshape(ns[i+1])
        index += ns[i + 1]

        param_list.append(W)
        param_list.append(b)

    return param_list


def wrap(param_list):
    """ Concatene les paramètres
    Cette fonction sert à concatener les tenseurs
    à chaque couche de réseau en un gros vecteur

    Parameters
    ----------
    param_list : list
        liste qui contient les tenseurs

    Returns
    -------
    torch.Tensor
        Le gros tenseur
    """
    listLen = len(param_list)

    theta = param_list[0].reshape(np.prod(param_list[0].size()), 1)

    for i in range(1, listLen):
        theta = torch.concat(
            (theta, param_list[i].reshape(np.prod(param_list[i].size()), 1)))

    return theta


class FNN(nn.Module):
    """ Definition d'un FNN
    """

    def __init__(self, ns, theta=None):
        """ Initialisation
        Construction d'une instance de la class FNN 
        à partir des paramètres données

        Parameters
        ----------
        ns : list 
            une liste de nombres de couches
        theta : torch.Tensor (None)
            une tenseur de paramètres

        Returns
        -------
        FNN
            L'instance de la classe
        """
        super(FNN, self).__init__()
        self.funclist = []
        self.nbLayers = len(ns)
        self.ns = ns
        for i in range(self.nbLayers-1):
            self.funclist.append(nn.Linear(ns[i], ns[i+1]))
        if (theta != None):
            self.update_weights(theta)

    def forward(self, x):
        """ Evaluation
        Cette fonction fait une forward pass du FNN

        Parameters
        ----------
        x : torch.Tensor
            la valeur d'entré
        Returns
        -------
        torch.Tensor
            Le résultat du forward pass
        """
        for i in range(self.nbLayers - 2):
            x = self.funclist[i](x)
            x = torch.tanh(x)
            # x = torch.relu(x)
        x = self.funclist[self.nbLayers - 2](x)
        return x

    def update_weights(self, thetas):
        """ Mets à jour les paramètres (poids)
        Remplace les poids courants

        Parameters
        ----------
        thetas : torch.Tensor
                liste qui contient les tenseurs

        Returns
        -------
        """
        param_list = segment(thetas, self.ns)
        nbparam = len(param_list)
        with torch.no_grad():
            for i in range(self.nbLayers - 1):
                self.funclist[i].weight = nn.Parameter(param_list[i*2])
                self.funclist[i].bias = nn.Parameter(param_list[i*2 + 1])

    def getTheta(self):
        """ renvoie les poids

        Parameters
        ----------

        Returns
        -------
        torch.Tensor
            Le gros tenseur de poids
        """
        param_list = []
        for i in range(self.nbLayers - 1):
            param_list.append(self.funclist[i].weight)
            param_list.append(self.funclist[i].bias)

        return wrap(param_list)


def modelSize(ns):
    """ calcul le nombre de poids

    Parameters
    ----------
    ns : list
        liste des nombres de neurones par couches
    Returns
    -------
    int
        Le nombre de poids
    """
    return np.sum([(ns[i]+1)*ns[i+1] for i in range(len(ns)-1)])


def plotTubeMean(XT, y, thetas, ns):
    """ plot un tube centré sur la moyenne
    Fait un forward pass avec en entrée XT pour chaque 
    theta dans theta, on evalue ensuite la moyenne
    la variance, les bornes moyenne +- 3 variances
    En on plot le fil de la moyenne, et le tude des deux bornes

    Parameters
    ----------
    XT : torch.Tensor
        tenseur d'entrée
    y : torch.Tensor 
        Solution bruité
    thetas : list of torch.Tensor
        un échantillion de poids
    ns : list 
        Le nombre de neurones par couche

    Returns
    -------
    """
    ymax = torch.max(y)
    ymin = torch.min(y)
    yspan = ymax - ymin

    N = thetas.shape[1]
    y_hats = torch.concat(
        tuple([FNN(ns, thetas[:, i]).forward(XT) for i in range(0, N)]), 1)

    std = torch.sqrt(y_hats.var(1))
    mean = y_hats.mean(1)

    binf = (mean - 3*std).reshape(y.shape).detach().numpy()
    bsup = (mean + 3*std).reshape(y.shape).detach().numpy()

    plt.fill_between(XT.ravel(), binf[:, 0],
                     bsup[:, 0], color='gray', label="[]")
    plt.plot(XT, (mean).reshape(y.shape).detach().numpy(),
             label="Mean", color="r")
    plt.scatter(XT, y, marker='+', color='k')
    plt.ylim(ymin - 0.01*yspan, ymax + 0.01*yspan)
    plt.legend()


def plotTubeMedian(XT, y, thetas, ns):
    """ plot un tube centré sur la médiane
    Fait un forward pass avec en entrée XT pour chaque 
    theta dans theta, on evalue ensuite la mediane
    le 1er et le 3ème quartile, les quantiles 2.5% et 99.75%
    En on plot le fil de la médiane, et le tube 1er et 3ème
    quartile et le tube quantile 2.5% et 99.75%

    Parameters
    ----------
    XT : torch.Tensor
        tenseur d'entrée
    y : torch.Tensor 
        Solution bruité
    thetas : list of torch.Tensor
        un échantillion de poids
    ns : list 
        Le nombre de neurones par couche

    Returns
    -------
    """

    ymax = torch.max(y)
    ymin = torch.min(y)
    yspan = ymax - ymin

    N = thetas.shape[1]
    y_hats = torch.concat(
        tuple([FNN(ns, thetas[:, i]).forward(XT) for i in range(0, N)]), 1)

    q2_5 = y_hats.quantile(0.025, 1).reshape(y.shape).detach().numpy()
    q97_5 = y_hats.quantile(0.975, 1).reshape(y.shape).detach().numpy()
    q25 = y_hats.quantile(0.25, 1).reshape(y.shape).detach().numpy()
    q75 = y_hats.quantile(0.75, 1).reshape(y.shape).detach().numpy()

    plt.fill_between(XT.ravel(), q2_5[:, 0],
                     q97_5[:, 0], alpha=0.5, color='gray')
    plt.fill_between(XT.ravel(), q25[:, 0], q75[:, 0],
                     alpha=0.5, color='k', label='IQR')
    plt.plot(XT, y_hats.quantile(0.5, 1).reshape(
        y.shape).detach().numpy(), label="Median", color='r')

    plt.scatter(XT, y, marker='+', color='k', label='Training data')
    plt.ylim(ymin - 0.01*yspan, ymax + 0.01*yspan)
    plt.legend()


def BnnAbcSs(N, l, P0, epsilon, sigma_0, fact, ns, XT, y):
    """ BNN_ABC_SS

    Parameters
    ----------
    N : int
    l : int
    P0 : float
    epsilon : float
    sigma_0 : float : 
    fact : float
    ns : list 
    XT : torch.Tensor
    y : torch.Tensor

    Returns
    -------
    thetas : list of torch.Tensor

    rho_n : torch.Tensor

    """
    NP0 = int(N*P0)
    invP0 = int(1/P0)
    lll = modelSize(ns)

    thetas = torch.randn(lll, N)

    y_hats = torch.concat(
        tuple([FNN(ns, thetas[:, i]).forward(XT) for i in range(0, N)]), 1)

    rho_n = torch.cdist(y_hats.t(), y.t())

    sigma_j = sigma_0

    l_eps = []
    for j in range(0, l):
        rho_n, indices = torch.sort(rho_n, 0)
        thetas = thetas[:, indices.t()[0]]

        epsilon_j = rho_n[int(N*P0)]

        # for debugging purposes
        l_eps.append(epsilon_j)

        thetasSeeds = (thetas[:, :NP0].t()).repeat(invP0, 1).t()
        rho_nSeeds = (rho_n[:NP0, 0].reshape(NP0, 1)).repeat(invP0, 1)

        # Resampling using seeds
        thetasResamples = torch.normal(thetasSeeds, sigma_j)
        # Evaluating performaces
        y_hatsResamples = torch.concat(
            tuple([FNN(ns, thetasResamples[:, i]).forward(XT) for i in range(0, N)]), 1)
        rho_nResamples = torch.cdist(y_hatsResamples.t(), y.t())

        # Mask creation
        mask = torch.diag((rho_nResamples <= rho_nSeeds)[:, 0].float())

        thetas = torch.matmul(thetasResamples, mask) + \
            torch.matmul(thetasSeeds, torch.eye(N) - mask)
        rho_n = torch.matmul(mask, rho_nResamples) + \
            torch.matmul(torch.eye(N) - mask, rho_nSeeds)

        sigma_j = sigma_j - 0.1

    return thetas, rho_n
