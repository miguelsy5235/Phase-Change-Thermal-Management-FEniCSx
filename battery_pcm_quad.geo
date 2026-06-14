// Battery + PCM structured quad mesh using 4 blocks

L = 0.10;

x1 = 0.035;
x2 = 0.065;
y1 = 0.040;
y2 = 0.060;

nx = 20;
ny = 20;

// Outer square
Point(1) = {0, 0, 0};
Point(2) = {L, 0, 0};
Point(3) = {L, L, 0};
Point(4) = {0, L, 0};

// Inner battery rectangle
Point(5) = {x1, y1, 0};
Point(6) = {x2, y1, 0};
Point(7) = {x2, y2, 0};
Point(8) = {x1, y2, 0};

// Outer boundary
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Battery boundary
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

// Connection lines from outer corners to battery corners
Line(9)  = {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

// Four PCM regions
Curve Loop(1) = {1, 10, -5, -9};
Plane Surface(1) = {1}; // bottom PCM

Curve Loop(2) = {2, 11, -6, -10};
Plane Surface(2) = {2}; // right PCM

Curve Loop(3) = {3, 12, -7, -11};
Plane Surface(3) = {3}; // top PCM

Curve Loop(4) = {4, 9, -8, -12};
Plane Surface(4) = {4}; // left PCM

// Battery region
Curve Loop(5) = {5, 6, 7, 8};
Plane Surface(5) = {5};

// Structured mesh
Transfinite Curve {1,3,5,7} = nx + 1;
Transfinite Curve {2,4,6,8} = ny + 1;

Transfinite Curve {9,10,11,12} = nx + 1;

Transfinite Surface {1};
Transfinite Surface {2};
Transfinite Surface {3};
Transfinite Surface {4};
Transfinite Surface {5};

Recombine Surface {1,2,3,4,5};

// Physical groups
Physical Surface("PCM", 1) = {1,2,3,4};
Physical Surface("Battery", 2) = {5};

Physical Curve("Walls", 3) = {1,2,3,4};
Physical Curve("BatteryWall", 4) = {5,6,7,8};
