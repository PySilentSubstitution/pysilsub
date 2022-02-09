Background
==========
*"Silent substitution ... involves using pairs of lights to selectively stimulate one class of retinal photoreceptor whilst maintaining a constant level of activation in others."*

*"The fundamental challange of silent substitution is in finding the settings for a multiprimary stimulation device to produce lights that selectively stimulate the photoreceptor(s) of interest. PySilSub is a Python software that aims to streamline this process."*
  
A normal human retina contains several types of photosensetive cell. Enabling colour vision at mesopic/photopic light levels are the short-, medium-, and long-wavelength-light-sensetive `cone photoreceptors <https://en.wikipedia.org/wiki/Cone_cell>`_. Cone cells are packed densley into the fovea and distributed sparsely outside of this area. For scotopic (twilight) vision we have `rod photoreceptors <https://en.wikipedia.org/wiki/Rod_cell>`_. Though not present at the fovea, rods are the most numerous of the photoreceptor cells and are otherwise widely distributed in retina. Finally, discovered in human retinae only at the turn of the millenium--`intrinsically photosensetive retinal ganglion cells (ipRGCs) <https://en.wikipedia.org/wiki/Intrinsically_photosensitive_retinal_ganglion_cell>`_ expressing the photopigment `melanopsin <https://en.wikipedia.org/wiki/Melanopsin>`_ in their axons and soma. ipRGCs do not contribute to vision in the same way as rods and cones but they do play important roles in 'non-visual' functions, such as circadian photoentraiment and pupil control, via direct projections to the suprachiasmatic nucleus of the hypothalamus and the olivery pretectal nucleus of the midbrain (`Gamlin et al., 2007 <https://doi.org/10.1016/j.visres.2006.12.015>`_; `Ruby et al., 2002 <https://www.science.org/doi/abs/10.1126/science.1076701>`_). 

.. image:: ../../img/eye_retina.png
  :alt: Eye, retina and photoreceptors.
  
As illustrated above, the photoreceptors have different spectral sensetivities. The curve for each type of photoreceptor essentially describes the probability of its capturing a photon at a given wavelength. Therefore S-cones are about 10 times more likely than L-cones to capture photons at 450 nm, and the liklihood of L- and M-cones capturing at 550 nm is about the same. Because the spectral sensetivities of the photoreceptors overlap, it should be clear that most lights in the visible spectrum will stimulate all types of photoreceptor, albeit to varying degrees. But what if we wanted to stimulate only one type of photoreceptor?

Silent substitution is an elegant solution to this problem. The technique involves using pairs of lights to selectively stimulate one class of retinal photoreceptor whilst maintaining a constant level of activation in--i.e., silencing--the others. This is possible owing to `Rushton's (1972) principle of univariance <https://en.wikipedia.org/wiki/Principle_of_univariance>`_, which states that the output of a photoreceptor is one-dimensional and depends upon quantum catch, not upon what quanta are caught. In other words, different light spectra will have an identical effect on a photoreceptor providing they lead to the same number of photons being absorbed. The prinicple of univariance and its relevance to silent substitution is covered in greater detail by `Estévez and Spekreijse (1982) <https://doi.org/10.1016/0042-6989(82)90104-3>`_ along with other details regarding the early history of the method. For a more recent review that includes a rundown of various methodological challanges, refer to `Spitschan and Woelders (2018) <https://doi.org/10.3389/fneur.2018.00941>`_.

In vision science, the method of silent substitution has contributed to our understanding of human colour vision mechanisms (e.g., `Horiguchi et al., 2013 <https://doi.org/10.1073/pnas.1214240110>`_) and it has enabled researchers to examine how targeted photoreceptor stimulation affects physiological responses such as melatonin suppression (e.g., `Allen et al., 2018 <https://doi.org/10.1093/sleep/zsy100>`_), the electroretinogram (e.g., `Maguire et al., 2017 <https://doi.org/10.1007/s10633-017-9571-4>`_) and the pupillary light reflex (e.g., `Spitschan et al., 2014 <https://doi.org/10.1073/pnas.1400942111>`_). Its potential as a diagnostic tool for retinal disease has also garnered attention in recent years (e.g., `Kuze et al., 2017 <https://doi.org/10.1016/j.optom.2016.07.004>`_; `Wise et al., 2021 <https://doi.org/10.1111/vop.12847>`_).

*PySilSub*
----------

The fundamental challange of silent substitution is in finding the settings for a multiprimary stimulation device to produce lights that selectively stimulate the photoreceptor(s) of interest. *PySilSub* is a Python software that aims to streamline this process. 

References
----------

- Allen, A. E., Hazelhoff, E. M., Martial, F. P., Cajochen, C., & Lucas, R. J. (2018). Exploiting metamerism to regulate the impact of a visual display on alertness and melatonin suppression independent of visual appearance. Sleep, 41(8), 1–7. https://doi.org/10.1093/sleep/zsy100

- Estévez, O., & Spekreijse, H. (1982). The “Silent Substitution” method in research. Vision Research, 22(6), 681–691. https://doi.org/10.1016/0042-6989(82)90104-3

- Gamlin, P. D. R., McDougal, D. H., Pokorny, J., Smith, V. C., Yau, K. W., & Dacey, D. M. (2007). Human and macaque pupil responses driven by melanopsin-containing retinal ganglion cells. Vision Research, 47(7), 946–954. https://doi.org/10.1016/j.visres.2006.12.015

- Horiguchi, H., Winawer, J., Dougherty, R. F., & Wandell, B. A. (2013). Human trichromacy revisited. Proceedings of the National Academy of Sciences of the United States of America, 110(3). https://doi.org/10.1073/pnas.1214240110

- Kuze, M., Morita, T., Fukuda, Y., Kondo, M., Tsubota, K., & Ayaki, M. (2017). Electrophysiological responses from intrinsically photosensitive retinal ganglion cells are diminished in glaucoma patients. Journal of Optometry, 10(4), 226–232. https://doi.org/10.1016/j.optom.2016.07.004

- Maguire, J., Parry, N. R. A., Kremers, J., Murray, I. J., & McKeefry, D. (2017). The morphology of human rod ERGs obtained by silent substitution stimulation. Documenta Ophthalmologica, 134(1), 11–24. https://doi.org/10.1007/s10633-017-9571-4

- Masland, R. H. (2001). The fundamental plan of the retina. Nature Neuroscience, 4(9), 877–886. https://doi.org/10.1038/nn0901-877

- Provencio, I., Rodriguez, I. R., Jiang, G., Hayes, W. P., Moreira, E. F., & Rollag, M. D. (2000). A novel human opsin in the inner retina. Journal of Neuroscience, 20(2), 600–605. https://doi.org/10.1523/jneurosci.20-02-00600.2000

- Reynolds, J. D., & Olitsky, S. E. (2011). Anatomy and phsiology of the retina. Pediatric Retina, 1–462. https://doi.org/10.1007/978-3-642-12041-1

- Ruby, N. F., Brennan, T. J., Xie, X., Cao, V., Franken, P., Heller, H. C., & O’Hara, B. F. (2002). Role of melanopsin in circadian responses to light. Science, 298(5601), 2211–2213. https://doi.org/10.1126/science.1076701

- Spitschan, M., Jain, S., Brainard, D. H., & Aguirre, G. K. (2014). Opponent melanopsin and S-cone signals in the human pupillary light response. Proceedings of the National Academy of Sciences of the United States of America, 111(43), 15568–15572. https://doi.org/10.1073/pnas.1400942111

- Spitschan, M., & Woelders, T. (2018). The method of silent substitution for examining melanopsin contributions to pupil control. Frontiers in Neurology, 9(NOV). https://doi.org/10.3389/fneur.2018.00941

- Wise, E. N., Foster, M. L., Kremers, J., & Mowat, F. M. (2021). A modified silent substitution electroretinography protocol to separate photoreceptor subclass function in lightly sedated dogs. Veterinary Ophthalmology, 24(1), 103–107. https://doi.org/10.1111/vop.12847

.. rubric:: Tables and indices
------------------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
