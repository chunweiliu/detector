function default_params = get_pyramid_params
% Return default parameters for getting feature pyramids. These parameters
% are subset of Tomasz Malisiewicz's definition.
default_params.detect_max_scale = 2.0;

% Levels-per-octave defines how many levels between 2x sizes in pyramid
% (denser pyramids will have more windows and thus be slower for 
% detection/training)
default_params.detect_levels_per_octave = 10;

% Initial parameters
init_params.features = @esvm_features;
init_params.sbin = 8;

% Not understanding why he store init_params in a field
default_params.init_params = init_params;