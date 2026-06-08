"""Funções de Privacidade Diferencial (mecanismo gaussiano).

Reúne o clipping de sensibilidade e o ruído gaussiano calibrado num só lugar,
para serem reaproveitados pelo cliente federado e pelos scripts de análise.
"""

import numpy as np


def achatar_parametros(coef, intercept):
    # Junta coef_ e intercept_ num único vetor 1D e guarda o corte para desfazer
    coef = np.asarray(coef, dtype=float)
    intercept = np.asarray(intercept, dtype=float)
    vetor = np.concatenate([coef.ravel(), intercept.ravel()])
    return vetor, coef.shape, intercept.shape


def restaurar_parametros(vetor, formato_coef, formato_intercept):
    # Desfaz o achatamento, devolvendo coef_ e intercept_ com as formas originais
    n_coef = int(np.prod(formato_coef))
    coef = vetor[:n_coef].reshape(formato_coef)
    intercept = vetor[n_coef:].reshape(formato_intercept)
    return coef, intercept


def clip_l2(vetor, clip_norm):
    # Limita a norma L2 a clip_norm (sensibilidade): reescala se passar, mantém se não
    norma = np.linalg.norm(vetor)
    if norma > clip_norm:
        vetor = vetor * (clip_norm / norma)
    return vetor


def calcular_sigma(clip_norm, epsilon, delta):
    # Desvio-padrão do mecanismo gaussiano: sigma = (C / epsilon) * sqrt(ln(1 / delta))
    return (clip_norm / epsilon) * np.sqrt(np.log(1.0 / delta))


def aplicar_dp(coef, intercept, clip_norm, epsilon, delta, rng=None):
    # Aplica clipping de sensibilidade e ruído gaussiano calibrado a coef_ e intercept_
    if rng is None:
        rng = np.random.default_rng()

    vetor, formato_coef, formato_intercept = achatar_parametros(coef, intercept)
    vetor = clip_l2(vetor, clip_norm)

    sigma = calcular_sigma(clip_norm, epsilon, delta)
    vetor = vetor + rng.normal(0.0, sigma, size=vetor.shape)

    coef_dp, intercept_dp = restaurar_parametros(vetor, formato_coef, formato_intercept)
    return coef_dp, intercept_dp, sigma
