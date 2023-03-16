===============
Cosmology
===============

The Friedmann equation is

.. math::
   \frac{H^2}{H_0^2} = \Omega_R a^{-4} + \Omega_M a^{-3} + \Omega_k a^{-2} + \Omega_\Lambda

where :math:`\Omega_R, \Omega_M, \Omega_\Lambda` are the present day values of
the radiation, matter, and dark energy (vacuum) density, respectively.  We
define the spatial curvature density :math:`\Omega_k = 1 - \Omega_0`, where
:math:`\Omega_0 = \Omega_R + \Omega_M + \Omega_\Lambda`.

.. math::
   H(a) = \frac{H_0}{a^2}\sqrt{\Omega_R + \Omega_M a + \Omega_k a^2 + \Omega_\Lambda a^4}

.. math::
   H = \frac{\dot{a}}{a} = \frac{da}{dt}\frac1a

.. math::
   dt = \frac{da}{aH}

.. math::
   t(a) = \int_0^a \frac{da'}{a'H(a')}

Let :math:`y^{2/3} = a'` then :math:`\frac23 y^{-1/3}dy = da'` and

.. math::
   t(a) = \int_0^a \frac23 y^{-1/3}dy\frac{1}{y^{2/3}H(y^{2/3})}
        = \int_0^a \frac23 \frac{dy}{yH(y^{2/3})}


The inverse function :math:`a(t)` is determine using a root-finding method.

====================
Drift-Kick Operators
====================

For each particle let :math:`\mathbf{r}` be the comoving position,
:math:`\mathbf{v}` the comoving velocity, :math:`\mathbf{a}` the comoving
acceleration, and :math:`\mathbf{p} = a^2\mathbf{v}` the conjugate momentum
which replaces the velocity variable.

The Drift update is defined as

.. math::
    \mathbf{r}(t_n+\Delta t)
    = \mathbf{r}(t_n) + \mathbf{p}(t_n) \int_{t_n}^{t_n + \Delta t} \frac{dt}{a^2}

and the Kick update is defined as

.. math::
    \mathbf{p}(t_n+\Delta t)
    = \mathbf{p}(t_n) + \mathbf{a}(t_n) \int_{t_n}^{t_n + \Delta t} \frac{dt}{a}


Using the substitution :math:`1/y = a'` and :math:`-y^{-2}dy = da` the
respective integrals can be rewritten as

.. math::
   \int_{t_n}^{t_n + \Delta t} \frac{dt}{a^2}
   = \int_{1/a(t_n)}^{1/a(t_n + \Delta t)} \frac{da}{aH}\frac{1}{a^2}
   = \int_{1/a(t_n)}^{1/a(t_n + \Delta t)} \frac{-y}{H(1/y)}da

.. math::
   \int_{t_n}^{t_n + \Delta t} \frac{dt}{a}
   = \int_{1/a(t_n)}^{1/a(t_n + \Delta t)} \frac{da}{aH}\frac{1}{a}
   = \int_{1/a(t_n)}^{1/a(t_n + \Delta t)} \frac{-1}{H(1/y)}da

====
TODO
====

#. List default values of density parameters and H0
#. Write down equations for special cases (no Lambda, e.g.).
