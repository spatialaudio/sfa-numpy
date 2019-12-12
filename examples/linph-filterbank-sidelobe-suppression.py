"""Sidelobe suppression with different equalization schemes

- The target magnitude responses fo the filter-bank is designed
  by using the zero-phase Butterworth responses.
  (not to confused with typical Butterworth filters)
- modal weight: 3D max-rE
- equalization
  (1) omnidirectional
  (2) diffuse-field
  (3) free-field

    Reference
    ---------
    Franz Zotter, "A linear-phase filter-bank approach to process
    rigid spherical microphone array recordings", in Proc. IcETRAN,
    Palic, Serbia, 2018. (see Fig. 8)
"""
import numpy as np
import matplotlib.pyplot as plt
from micarray.util import db, point_spread

N = 4
azi = np.linspace(0, np.pi, num=360)
equalization = ['omni', 'diffuse', 'free']

# Plot
rticks = np.arange(-36, 12, 12)
rlim = -36, 3
rticklabels = ['-36', '-24', '-12', '0 dB']
nn = np.arange(N+1)

fig, ax = plt.subplots(figsize=(10, 6), ncols=3, subplot_kw={'polar': True},
                       gridspec_kw={'wspace': 0.3})
for axi, eq in zip(ax, equalization):
    ps = np.stack([np.sum(point_spread(n, azi, equalization=eq), axis=0)
                  for n in range(N+1)])
    ps *= 1 / np.max(ps)
    axi.plot(azi[:, np.newaxis] * (-1)**nn, db(ps.T), lw=3, alpha=0.5)
    axi.set_title('{}'.format(eq), y=1.2)
    axi.set_rlabel_position(135)
    axi.set_rticks(rticks)
    axi.set_rlim(*rlim)
    axi.set_yticklabels(rticklabels)
for axi in ax[1:]:
    axi.set_xticklabels([])
ax[-1].legend(nn, title='subband', bbox_to_anchor=(1.1, 1))
plt.savefig('./sidelobe-suppression.png', bbox_inches='tight')
