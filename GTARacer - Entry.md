
```mermaid
classDiagram


class IOController{
    <<First Draft>>
    process_new_image(screenshot)
    fcw_subsumption
    speed_estimator
}


class FCWSubsumption{
    prediction_threshold
    fcw_estimator
    subsume(image_sequence)
}
IOController --> FCWSubsumption : contains


class Model{
    load_path
    parameters
    save(path)
}


class FCWEstimator{
    sequence_length
    estimate_collision_probability(image_sequence)
}
FCWSubsumption --> FCWEstimator : contains
Model *-- FCWEstimator


class SpeedEstimator{
    sequence_length
    estimate_speed(image_sequence)
}
Model *-- SpeedEstimator
IOController --> SpeedEstimator : contains


class MPCPathFollower{
    state_dimension
}


class PathFollower{
    <<First Draft>>
    get_path(image)
}
IOController *-- PathFollower


class RangeBearing{
    pid_controller
}
IOController *-- RangeBearing


class PIDPathFollower{
    <<First Draft>>
    pid_controller
}
PathFollower *-- PIDPathFollower


class PIDController{
    <<First Draft>>
    pid_constants
    get_control(error)
}
RangeBearing --> PIDController : contains
PIDPathFollower --> PIDController : contains


class MPCPathFollower
PathFollower *-- MPCPathFollower


class Pather{
    <<First Draft>>
}
PIDPathFollower --> Pather : contains


class PolyFitter{
    <<First Draft>>
}
Pather *-- PolyFitter


class OptimalFitter{
    <<First Draft>>
}
Pather *-- OptimalFitter


class score_path{
    <<First Draft>>
}
score_path o-- OptimalFitter : calls
MPCPathFollower --o score_path : calls


class WeightedFilter{
    <<First Draft>>
    weighted_filters
    filter(image)
}
RangeBearing --> WeightedFilter : contains
PolyFitter --> WeightedFilter : contains
OptimalFitter --> WeightedFilter : contains

class BinarySegmentation{
    <<First Draft>>
    segment(image)
}
WeightedFilter --> "multiple" BinarySegmentation : contains


class ColorRadiusSegmentation{
    <<First Draft>>
    target_color
    max_radius
}
BinarySegmentation *-- ColorRadiusSegmentation


class TrainedBinarySegmentation{
    model
}
BinarySegmentation *-- TrainedBinarySegmentation
```

- [ ] Add critical attributes/methods to the outPath above.
- [ ] Rethink the critical optimal pathing and optimal control parts of the design.
- [ ] Devise a way to define configurations and propagate parameters through the whole program, perhaps with a shared dictionary or named tuple.
- [ ] Do initial breaking changes on the library end, starting with the classes marked `<<First Draft>>`.
- [ ] Do initial breaking changes on the user end.
- [ ] Work from both ends until refactor is done.
- [ ] Use [crash sound effects](https://blog.gingerbeardman.com/2023/08/13/automatically-classifying-the-content-of-sound-files-using-ml/) to do RL.

`weighted_filters` will be something like

```python
weighted_filters = {
    'and': [
        (1.0, ColorRadiusSegmentation(waypoint_color, default_color_radius)),
        (0.1, ColorRadiusSegmentation(waypoint_color, default_color_radius)),
    ]
}
```

The idea is that this sets us up for making a costmap for optimal pathing. But maybe it's simultaneously too complex (will I ever need an 'or'?) and not complex enough (how can I handle selectively smoothing filters?).
