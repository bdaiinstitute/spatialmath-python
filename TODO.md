# Work still to do for spatialmath-python


## classes
* Empty constructor for all
* == != for all classes
* UnitQuaternion
  * angle
  * vectorization of vec3 multiplication, v3mul

* norm, print, plot for all classes
* use @ operator for composition with normalization
* vectorization of RPY, Eul, AngVec constructors and converters
* add from/to to transformations, add check that multiplication is valid
  * keep class variables for a dict of all poses
  * use dot to make a graph
  * use Unicode sub/superscripts to format this nicely (not all letters possible)
* UnitQuaternion interp, vectorize

## low level

* delta2tr, tr2delta
* colvec to vectors.py
* trnorm
* trlog2
* trinterp
* homtrans
* animation
  * animate STL shape
  * animation, animation2 classes
* move skew/vex family to transforms[23]d
* tests for argcheck

## doco
* bring all parts to same level
  * pose2d, quaternion, tranforms2d
* include graphics in the doco
  * docs/source/figs
* complete Python notebook example for SE3 at least


## new work
* Twist class
* Plucker class
* symbolics as much as possible

## overall
* add support channels to doco
* cleanup all file headers
  * just the purpose
  * acknowledgement/history to comments
* remove assertions, replace with ValueError
* autopep8
* linting
* revise all error messages, more consistent

## tests
* add assertRaises(except, func) to unit tests
* fix all TODOs
* argcheck tests

## other



* Travis matrix test for different Python/numpy versions
* fixup consequences of spatialmath-matlab name change
  * petercorke.com download page
  * Travis build for RTB matlab
  * MathWorks file exchange

