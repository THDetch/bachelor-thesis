\begin{itemize}
    \item SAR-to-Optical Image Translation and Cloud Removal Based on Conditional Generative Adversarial Networks: Literature Survey, Taxonomy, Evaluation Indicators, Limits and Future Directions
    \url{https://www.mdpi.com/2072-4292/15/4/1137}

    \item UnCRtainTS: Uncertainty Quantification for Cloud Removal in Optical Satellite Time Series
    \url{https://arxiv.org/abs/2304.05464}

    \item DiffCR: A Fast Conditional Diffusion Framework for Cloud Removal from Optical Satellite Images
    \url{https://arxiv.org/abs/2308.04417}

    \item Conditional Diffusion for SAR to Optical Image Translation
    \url{https://ieeexplore.ieee.org/document/10330015}

    \item Cloud Removal for Remote Sensing Imagery via Spatial Attention Generative Adversarial Network
    \url{https://arxiv.org/abs/2009.13015}

    \item SAR to Optical Image Translation with Color Supervised Diffusion Model
    \url{https://arxiv.org/abs/2407.16921}

    \item SAR-to-optical image translation by a variational generative adversarial network
    \url{https://www.tandfonline.com/doi/full/10.1080/2150704X.2022.2068986}

    \item SAR-to-Optical Image Translation Using Vision Transformer-Based CGAN
    \url{https://ieeexplore.ieee.org/document/10948876}

    \item Cloud Removal in Remote Sensing Images Using Generative Adversarial Networks and SAR-to-Optical Image Translation
    \url{https://ieeexplore.ieee.org/document/9627647}

    \item A Conditional Generative Adversarial Network to Fuse Sar And Multispectral Optical Data For Cloud Removal From Sentinel-2 Images
    \url{https://ieeexplore.ieee.org/document/8519215}

    \item Assessing the Potential of Multi-Temporal Conditional Generative Adversarial Networks in SAR-to-Optical Image Translation for Early-Stage Crop Monitoring
    \url{https://www.mdpi.com/2072-4292/16/7/1199}

    \item High-Resolution SAR-to-Multispectral Image Translation Based on S2MS-GAN
    \url{https://www.mdpi.com/2072-4292/16/21/4045}

    \item GAN generation of synthetic multispectral satellite images
    \url{https://www.researchgate.net/publication/346821973_GAN_generation_of_synthetic_multispectral_satellite_images}

    \item SEN12MS – A CURATED DATASET OF GEOREFERENCED MULTI-SPECTRAL SENTINEL-1/2 IMAGERY FOR DEEP LEARNING AND DATA FUSION
    \url{https://isprs-annals.copernicus.org/articles/IV-2-W7/153/2019/}

    \item Cloud removal in Sentinel-2 imagery using a deep residual neural network and SAR-optical data fusion
    \url{https://www.sciencedirect.com/science/article/pii/S0924271620301398}

    \item THE SEN1-2 DATASET FOR DEEP LEARNING IN SAR-OPTICAL DATA FUSION
    \url{https://isprs-annals.copernicus.org/articles/IV-1/141/2018/}
\end{itemize}
Bachelor Thesis in the Field of Remote Sensing
Topic:
Generation of optical remote sensing data (Sentinel-2) based on radar remote sensing data (Sentinel-1) using generative AI.

Background:
Remote sensing (RS) collects data of the Earth’s surface using various sensors mounted on different airborne or spaceborne platforms. These can include satellites, the International Space Station (ISS), as well as manned aircraft or drones. Remote sensing data provide important information about our environment. A wide range of applications is covered through different missions, which vary mainly by platform and sensor type, each with their own strengths and weaknesses depending on the intended mission and use case.

Since the 2000s, RS data archives have grown enormously. Today, the exact global data volume is difficult to estimate, but it likely lies between several hundred petabytes and a few exabytes. Nevertheless, for certain applications, suitable data are often unavailable or insufficient—e.g., due to persistent cloud cover, sensor failure or malfunction, or because no matching data were acquired at the desired point in time.

Against this background, it would be desirable and useful to have methods (e.g., using generative AI) that can artificially generate suitable data for specific applications based on existing but otherwise unsuitable RS data. Approaches to this problem are already present in the literature (e.g., Zhao et al., 2022; Patel 2023; Poornima et al. 2023; Pan 2020).

Description:
As part of this bachelor thesis, a method will first be developed that can generate optical Sentinel-2 data from Sentinel-1 radar data using generative AI. For this purpose, the freely accessible archives of the Copernicus network are available, whose data can also be accessed automatically via API.

Within the KIWA research project (https://www.kiwa-projekt.de/
), Sentinel-2 data are automatically retrieved from the archives and used to identify forest fire areas and their structure, in order to determine potentially particularly vulnerable forest areas. If the newly developed generative AI method is successfully validated, it can be integrated into the KIWA workflow to potentially close existing data gaps.

In a similar way, further bachelor theses in this context could deal with generating unavailable but needed data from existing standard RS datasets.

Literature:

Zhao, J. et al. (2022). “SAR-to-optical image translation by a variational generative adversarial network.” Remote Sensing Letters, 13(7), 672–682. https://doi.org/10.1080/2150704X.2022.2068986

Pan, H. (2020). “Cloud Removal for Remote Sensing Imagery via Spatial Attention Generative Adversarial Network.” arXiv Preprint arXiv:2009.13015.

Patel, N. (2023). “Generative Artificial Intelligence and Remote Sensing: A Perspective on the Past and the Future.” IEEE Geoscience and Remote Sensing Magazine 11 (2): 86–100.

Poornima, E. et al. (2023). “Deep Generative Models for Automated Dehazing Remote Sensing Satellite Images.” In E3S Web of Conferences, 430:01024. EDP Sciences.
