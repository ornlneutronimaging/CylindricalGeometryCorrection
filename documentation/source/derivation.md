# Cylindrical Geometry Correction for Transmission Data

```{important}
**Status: design document, not yet implemented.** This page derives the
Beer-Lambert (exponential) correction with per-TOF μ estimation. The
correction currently implemented in `neutron_geomcorr` is a **linear
chord-length division** (see {doc}`introduction`), which is valid only for
inputs proportional to path length (attenuation, −ln(T), or thickness maps)
— not for raw transmission data. Reconciling the two — implementing this
derivation or re-scoping the package documentation — is tracked in
[issue #57](https://github.com/ornlneutronimaging/CylindricalGeometryCorrection/issues/57).
The algorithm below is prototyped in `notebooks/00_development.ipynb`.
```

This document presents the methodology for applying cylindrical geometry correction to transmission data, specifically in the context of neutron or X-ray imaging where Beer-Lambert law governs the attenuation of intensity through a material.

---

## Background: Beer-Lambert Law for Transmission

The transmitted intensity $ I(x, y) $ at a position $(x, y)$ through a material is described by the Beer-Lambert law:

$$
I(x, y) = I_0 \cdot e^{-\mu \cdot L(x, y)}
$$

where:

- $ I(x, y) $ is the transmitted intensity at position $(x, y)$,
- $ I_0 $ is the incident (background) intensity,
- $ \mu $ is the linear attenuation coefficient (accounting for both absorption and scattering),
- $ L(x, y) $ is the chord length through the material at position $(x, y)$.

For a vertical cylindrical sample, the system can be simplified by considering only the horizontal coordinate $ x $, assuming uniformity along $ y $:

$$
I(x) = I_0 \cdot e^{-\mu \cdot L(x)}
$$

where $ L(x) $ is the chord length through the cylinder at position $ x $.

---

### Transmission and Attenuation Coefficient

The normalized transmission $ T(x) $, also known as the normalized radiograph, is defined as the ratio of transmitted intensity to incident intensity:

$$
T(x) = \frac{I(x)}{I_0} = e^{-\mu \cdot L(x)}
$$

From this relation, the attenuation coefficient $ \mu $ can be calculated as:

$$
\mu = - \frac{\ln(T(x))}{L(x)}
$$

---

## Practical Considerations

In practice, ideal background measurements $ I_0 $ may not be available or may be imperfect, resulting in transmission values $ T(x) > 1 $, which leads to negative or non-physical values for $ \mu $. To address this, two robust methods for estimating $ \mu $ are presented below.

---

### Method 1: Discrete Method (Two-Column Approach)

This method leverages transmission values at two distinct positions $ x_1 $ and $ x_2 $ on the cylinder to estimate a physically meaningful attenuation coefficient.

Given:

$$
T(x_1) = e^{-\mu \cdot L(x_1)} \quad \text{and} \quad T(x_2) = e^{-\mu \cdot L(x_2)}
$$

Taking the ratio:

$$
\frac{T(x_1)}{T(x_2)} = \frac{e^{-\mu \cdot L(x_1)}}{e^{-\mu \cdot L(x_2)}} = e^{-\mu \cdot (L(x_1) - L(x_2))}
$$

Solving for $ \mu $:

$$
\mu = -\frac{1}{L(x_2) - L(x_1)} \cdot \ln\left(\frac{T(x_1)}{T(x_2)}\right)
$$

**Interpretation:** This approach ensures that the calculated attenuation coefficient $ \mu $ is always positive and physically meaningful, as it relies on relative transmission differences rather than absolute values. The choice of $ x_1 $ and $ x_2 $ should correspond to positions with reliable transmission data.

---

### Method 2: Iterative Method (Global Regression with Nuisance Term)

An alternative approach involves performing a global regression across all spatial positions $ x $ to estimate $ \mu $ while accounting for nuisance parameters (such as background fluctuations or instrumental effects). This iterative method fits the model:

$$
T(x) \approx e^{-\mu \cdot L(x)} \cdot N(x)
$$

where $ N(x) $ represents a nuisance term that captures deviations from ideal background normalization.

The regression optimizes the parameters $ \mu $ and $ N(x) $ to minimize the discrepancy between measured and modeled transmissions across the entire dataset. This method is particularly useful when background measurements are noisy or non-uniform.

---

### Correction Application per Wavelength (TOF Bin)

It is important to note that the attenuation coefficient $ \mu $ and the corresponding correction factor are wavelength-dependent (i.e., vary with Time-of-Flight (TOF) bins). Therefore, the correction must be applied separately for each wavelength bin to accurately account for energy-dependent attenuation.

---

## Cylindrical Sample Thickness Correction

Once $ \mu $ is determined, the goal is to transform the measured transmission $ T(x) $ corresponding to the chord length $ L(x) $ at position $ x $ to a uniform thickness corresponding to the cylinder diameter $ D $.

Given:

$$
T(x) = e^{-\mu \cdot L(x)}
$$

We define the corrected transmission for uniform thickness $ D $ as:

$$
T^D(x) = e^{-\mu \cdot D}
$$

The correction factor $ C(x) $ to be applied to the measured transmission is:

$$
C(x) = \frac{T^D(x)}{T(x)} = e^{-\mu \cdot (D - L(x))}
$$

This factor adjusts the transmission data as if the sample thickness were uniform and equal to the diameter $ D $.

> **Note:** Since the attenuation coefficient $ \mu $ is derived using differences in chord lengths, the correction factor $ C(x) $ is dimensionless and physically consistent.

---

This methodology provides a robust framework for correcting cylindrical sample transmission data, enabling accurate attenuation coefficient estimation and uniform thickness correction across all wavelengths.
