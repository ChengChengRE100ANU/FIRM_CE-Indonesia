"""
The fast_methods are functions that effectively take the place of methods for the jitclasses. These 'pseudo-methods' are
used so that the fastmath=True argument can be provided to the njit decorator, which substantially improves optimisation
time. At this stage, fastmath cannot be used directly with the jitclass decorator or methods within a jitclass.

Each jitclass has its own fast_methods module. Each pseudo-method takes an instance of the jitclass as the first argument. The
pseudo-methods predominantly modify the jitclass instances through side-effects.
"""
