from collections import OrderedDict

# Define thresholds
thresholds = dict(
    polynomial_fn=
                {

                 'exp1': OrderedDict(
                     GeDI=0.7,
                     mean_squared_error=0.7,
                 ),

                'exp2': OrderedDict(
                     GeDI=0.5,
                     mean_squared_error=0.5,
                 ),

                 'exp3': OrderedDict(
                     GeDI=0.2,
                     mean_squared_error=0.2,
                ),
            },
    group_weights=
            {

                'exp1': OrderedDict(
                    GeDI=0.7,
                    score_absolute_error=0.7,
                ),

                'exp2': OrderedDict(
                    GeDI=0.5,
                    score_absolute_error=0.5,
                ),

                'exp3': OrderedDict(
                    GeDI=0.2,
                    score_absolute_error=0.2,
                ),
            }
)
