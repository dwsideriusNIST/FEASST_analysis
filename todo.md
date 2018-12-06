## TO DO:
1. Decide better way to store moments of N and U; probably a list structure
1. Institute a check against temperature extrapolation when higher moments of (canonical) U(N) are not available
1. Decide how to store properties: probably output in a dictionary (cf. how to store moments)
1. Additional Methods:
 + equilibrium condition
 + spinodal conditions?
 + Extrema?
 + Entropy functions? E.g. canonical entropy, grand-canonical entropy, etc.
1. Decide what to do with "order" parameter for argrelextrema
1. Look into "watershed" algorithm
1. Keep the main class pretty simple; don't build in too many functions; may even consider taking the "properties" method outside the main class.
1. Can we request specific properties via a passed function, rather than hard-coding the property computations?
