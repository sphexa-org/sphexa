# Visualization

- [Catalyst](https://catalyst-in-situ.readthedocs.io/en/latest/introduction.html)
    - [ParaViewCatalyst](https://docs.paraview.org/en/latest/Catalyst/index.html)
    - [AscentCatalyst](https://github.com/Alpine-DAV/ascent/blob/develop/src/libs/catalyst/AscentCatalyst.cxx)
    - [AdiosCatalyst](https://gitlab.kitware.com/paraview/adioscatalyst)
- [Ascent](https://ascent.readthedocs.io/en/latest/index.html)

These means that it is possible to setup any visualiation paradigm:

- post-hoc (e.g. using the stub implementation or adioscatalyst to bp5)
- in-situ, with any of the solution above
- in-transit with AdiosCatalyst
- hybrid (in-situ + in-transit) with AdiosCatalyst

## Catalyst

```
--catalyst <file>
```

where file, depending on the catalyst implementation chosen at runtime, can be either:

- a paraview catalyst python script
- an ascent script
- an adios config file (note: it must have .xml extension)

## Ascent

```
--ascent <ascent_config_file>
```

# Known Limitations

## Catalyst

### AscentCatalyst

- 'grid' name is hard-coded

### AdiosCatalyst

- GPU support is WIP in [PR](https://gitlab.kitware.com/paraview/adioscatalyst/-/merge_requests/40)

## Ascent

In 0.9.6 some fixes have been added:

- add support for implicit topology in GPU expressions (https://github.com/Alpine-DAV/ascent/pull/1677)
- fix performance problem with unstructured topology (https://github.com/Alpine-DAV/ascent/pull/1679)
