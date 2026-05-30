import logging

import torch

from saicinpainting.evaluation.evaluator import InpaintingEvaluatorOnline, ssim_fid100_f1
from saicinpainting.evaluation.losses.base_loss import SSIMScore
from netcdf.evaluation import (
    RMSEScore, CorrelationScore, StabilityViolationScore,
    BBLGradientScore, DomainRMSEScore, OIBaselineScore,
)


def make_evaluator(kind='default', ssim=True,
                   rmse=False, correlation=False,
                   stability_violation=False, bbl_gradient=False,
                   domain_rmse=False, oi_baseline=False,
                   integral_kind=None, **kwargs):
    logging.info('Make evaluator %s', kind)
    metrics = {}
    if ssim:
        metrics['ssim'] = SSIMScore()
    if rmse:
        metrics['rmse'] = RMSEScore()
    if correlation:
        metrics['correlation'] = CorrelationScore()
    if stability_violation:
        metrics['stability_violation'] = StabilityViolationScore()
    if bbl_gradient:
        metrics['bbl_gradient'] = BBLGradientScore()
    if domain_rmse:
        metrics['domain_rmse'] = DomainRMSEScore()
    if oi_baseline:
        metrics['oi_baseline'] = OIBaselineScore()

    if integral_kind is None:
        integral_func = None
    elif integral_kind == 'ssim_fid100_f1':
        integral_func = ssim_fid100_f1
    else:
        raise ValueError('Unexpected integral_kind=%s' % integral_kind)

    if kind == 'default':
        return InpaintingEvaluatorOnline(scores=metrics,
                                         integral_func=integral_func,
                                         integral_title=integral_kind,
                                         **kwargs)
