# -*- coding: utf-8 -*-
"""
Corpus of text chunks and test questions from
"Introduction to Calculus Volume II" by J.H. Heinbockel.
Used for experimental comparison of RAG vs Onto-RAG.
"""


def get_corpus():
    """Returns a list of text chunks from the Calculus textbook."""
    return [
        # -- Limits and Continuity -------------------------------------------
        "A limit describes the value that a function approaches as the input "
        "approaches a particular point. The formal epsilon-delta definition "
        "states that for every epsilon greater than zero there exists a delta "
        "greater than zero such that whenever the distance between x and the "
        "limit point is less than delta, the distance between f(x) and L is "
        "less than epsilon. Limits are foundational to all of calculus.",

        "Continuity of a function at a point means that the limit of the "
        "function as x approaches that point equals the function value at "
        "that point. A function is continuous on an interval if it is "
        "continuous at every point in that interval. The Intermediate Value "
        "Theorem states that a continuous function on a closed interval "
        "takes on every value between f(a) and f(b).",

        # -- Derivatives -----------------------------------------------------
        "The derivative of a function f(x) is defined as the limit of the "
        "difference quotient: f'(x) = lim(h->0) [f(x+h) - f(x)] / h. "
        "It represents the instantaneous rate of change of the function "
        "and gives the slope of the tangent line at any point on the curve. "
        "The process of finding derivatives is called differentiation.",

        "The Chain Rule states that the derivative of a composite function "
        "f(g(x)) equals f'(g(x)) multiplied by g'(x). This rule is essential "
        "for differentiating nested functions. For example, the derivative of "
        "sin(x^2) is cos(x^2) * 2x. The Chain Rule extends naturally to "
        "compositions of three or more functions.",

        "The Product Rule states that the derivative of a product of two "
        "functions u(x) and v(x) is u'(x)v(x) + u(x)v'(x). The Quotient "
        "Rule gives the derivative of u/v as [u'v - uv'] / v^2. Both rules "
        "are derived from the definition of the derivative using limits.",

        "Higher-order derivatives are obtained by differentiating repeatedly. "
        "The second derivative f''(x) represents the rate of change of the "
        "rate of change, which geometrically describes the concavity of the "
        "curve. If f''(x) > 0, the function is concave up; if f''(x) < 0, "
        "it is concave down. Points where concavity changes are called "
        "inflection points.",

        # -- Integrals -------------------------------------------------------
        "The definite integral of f(x) from a to b represents the signed "
        "area under the curve y = f(x) between x = a and x = b. It is "
        "defined as the limit of Riemann sums. The integral is computed by "
        "partitioning the interval [a,b] into subintervals and summing the "
        "products of function values and subinterval widths as the partition "
        "becomes infinitely fine.",

        "The Fundamental Theorem of Calculus connects differentiation and "
        "integration. Part 1 states that if F(x) = integral from a to x of "
        "f(t)dt, then F'(x) = f(x). Part 2 states that the definite integral "
        "of f from a to b equals F(b) - F(a), where F is any antiderivative "
        "of f. This theorem is the cornerstone of integral calculus.",

        "Integration by parts is a technique derived from the Product Rule. "
        "It states that the integral of u dv equals uv minus the integral "
        "of v du. This method is particularly useful for integrating products "
        "of polynomial and exponential or trigonometric functions. Repeated "
        "application may be necessary for some integrals.",

        "Integration by substitution, also known as u-substitution, is the "
        "reverse of the Chain Rule. If the integrand can be written as "
        "f(g(x)) * g'(x), then substituting u = g(x) simplifies the "
        "integral to the integral of f(u) du. This technique is one of the "
        "most commonly used methods for evaluating integrals.",

        # -- Series ----------------------------------------------------------
        "An infinite series is the sum of infinitely many terms a_1 + a_2 + "
        "a_3 + ... A series converges if its sequence of partial sums "
        "approaches a finite limit. The geometric series sum(r^n) converges "
        "to 1/(1-r) when |r| < 1. Convergence tests include the ratio test, "
        "root test, comparison test, and integral test.",

        "A Taylor series represents a function as an infinite sum of terms "
        "calculated from the values of its derivatives at a single point. "
        "The Taylor series of f(x) around x = a is given by "
        "sum [f^(n)(a)/n!] * (x-a)^n. When a = 0, it is called a Maclaurin "
        "series. Common examples include e^x = sum(x^n/n!), "
        "sin(x) = sum((-1)^n * x^(2n+1)/(2n+1)!), and "
        "cos(x) = sum((-1)^n * x^(2n)/(2n)!).",

        "A power series is sigma from n=0 to infinity of c_n * (x-a)^n. "
        "Each power series has a radius of convergence R such that the "
        "series converges for |x-a| < R and diverges for |x-a| > R. "
        "Within the radius of convergence, a power series can be "
        "differentiated and integrated term by term.",

        # -- Partial Derivatives ---------------------------------------------
        "A partial derivative of a multivariable function f(x,y) with "
        "respect to x is the derivative taken while holding y constant, "
        "denoted df/dx or f_x. Similarly, df/dy holds x constant. "
        "Mixed partial derivatives f_xy and f_yx are equal for functions "
        "with continuous second-order partial derivatives, by Clairaut's "
        "theorem (also called Schwarz's theorem).",

        "The gradient of a scalar function f(x,y,z) is the vector of its "
        "partial derivatives: grad(f) = (df/dx, df/dy, df/dz). The gradient "
        "points in the direction of steepest ascent of the function and its "
        "magnitude gives the rate of maximum increase. It is perpendicular "
        "to the level surfaces of the function.",

        "The total differential of f(x,y) is df = (df/dx)dx + (df/dy)dy. "
        "This approximates the change in f when x and y change by small "
        "amounts dx and dy. The total derivative extends the Chain Rule "
        "to multivariable functions and is essential for understanding "
        "how composite multivariable functions change.",

        # -- Multiple Integrals ----------------------------------------------
        "A double integral of f(x,y) over a region R computes the volume "
        "under the surface z = f(x,y) above the region R in the xy-plane. "
        "It is evaluated as an iterated integral: integral(integral f(x,y) "
        "dA), where the order of integration can often be changed using "
        "Fubini's theorem, provided f is continuous on R.",

        "A triple integral extends double integrals to three dimensions, "
        "computing quantities like mass and volume over a three-dimensional "
        "region. Triple integrals can be evaluated in Cartesian, cylindrical, "
        "or spherical coordinates, with appropriate Jacobian determinants "
        "for coordinate transformations.",

        "The Jacobian determinant is used when changing variables in "
        "multiple integrals. For a transformation from (u,v) to (x,y), "
        "the Jacobian J = d(x,y)/d(u,v) = (dx/du)(dy/dv) - (dx/dv)(dy/du). "
        "The area element transforms as dA = |J| du dv. In polar coordinates, "
        "the Jacobian is r, so dA = r dr d(theta).",

        # -- Vector Calculus -------------------------------------------------
        "The divergence of a vector field F = (P, Q, R) is a scalar: "
        "div(F) = dP/dx + dQ/dy + dR/dz. It measures the rate at which "
        "the field spreads out from a point. A positive divergence indicates "
        "a source, while negative divergence indicates a sink. If div(F) = 0 "
        "everywhere, the field is called solenoidal or incompressible.",

        "The curl of a vector field F = (P, Q, R) is a vector: "
        "curl(F) = (dR/dy - dQ/dz, dP/dz - dR/dx, dQ/dx - dP/dy). "
        "The curl measures the rotation or circulation tendency of the field "
        "at a point. If curl(F) = 0, the field is irrotational, which implies "
        "that F is the gradient of some scalar potential function.",

        "A line integral of a vector field F along a curve C computes "
        "the work done by the field along the path. It is written as "
        "integral_C F . dr. If F is conservative (curl F = 0), the line "
        "integral depends only on the endpoints, not the path. A scalar "
        "potential phi exists such that F = grad(phi) for conservative fields.",

        "A surface integral of a vector field F over a surface S computes "
        "the flux of the field through the surface: integral_S F . n dS, "
        "where n is the unit outward normal. Surface integrals are used "
        "extensively in physics, for example in computing electric flux "
        "through a closed surface in Gauss's law of electrostatics.",

        # -- Fundamental Theorems --------------------------------------------
        "Green's Theorem relates a line integral around a simple closed "
        "curve C in the plane to a double integral over the region D "
        "enclosed by C. It states: integral_C (P dx + Q dy) = "
        "integral_D (dQ/dx - dP/dy) dA. Green's Theorem is a special "
        "two-dimensional case of Stokes' Theorem.",

        "Stokes' Theorem generalizes Green's Theorem to three dimensions. "
        "It relates the surface integral of the curl of a vector field "
        "over a surface S to the line integral of the field around the "
        "boundary curve of S: integral_S curl(F) . dS = integral_C F . dr. "
        "This theorem connects circulation and curl.",

        "The Divergence Theorem (Gauss's Theorem) relates the flux of a "
        "vector field through a closed surface to the volume integral of "
        "the divergence over the enclosed region: integral_S F . n dS = "
        "integral_V div(F) dV. This theorem is fundamental in fluid "
        "mechanics, electromagnetism, and heat transfer.",

        # -- Differential Equations ------------------------------------------
        "An ordinary differential equation (ODE) is an equation involving "
        "an unknown function y(x) and its derivatives. A first-order ODE "
        "has the form dy/dx = f(x,y). The solution is a function y(x) that "
        "satisfies the equation. Initial value problems specify y at a "
        "particular point to determine a unique solution.",

        "A separable ODE can be written as dy/dx = g(x) * h(y), which is "
        "solved by separating variables: integral dy/h(y) = integral g(x)dx. "
        "Linear first-order ODEs have the form dy/dx + P(x)y = Q(x) and "
        "are solved using an integrating factor mu(x) = exp(integral P(x)dx). "
        "The integrating factor method transforms the equation into an "
        "exact derivative.",

        # -- Applications ----------------------------------------------------
        "Optimization problems use derivatives to find maximum and minimum "
        "values of functions. Critical points occur where f'(x) = 0 or "
        "f'(x) is undefined. The First Derivative Test uses sign changes of "
        "f'(x) to classify critical points. The Second Derivative Test uses "
        "f''(x): if f''(c) > 0 the point is a local minimum, if f''(c) < 0 "
        "it is a local maximum.",

        "Related rates problems involve finding the rate of change of one "
        "quantity in terms of the rate of change of another. These problems "
        "use the Chain Rule applied to equations relating two or more "
        "variables that change with time. Common examples include expanding "
        "volumes, changing distances, and flowing liquids.",

        "Applications of integration include computing areas between curves, "
        "volumes of solids of revolution (using the disk, washer, or shell "
        "methods), arc length of curves, surface area of surfaces of "
        "revolution, and physical quantities like work, center of mass, "
        "and moments of inertia.",

        # -- Connections and Cross-references --------------------------------
        "The Fundamental Theorem of Calculus, Green's Theorem, Stokes' "
        "Theorem, and the Divergence Theorem are all manifestations of a "
        "single generalized Stokes' theorem. Each relates an integral over "
        "a boundary to an integral over the interior. This unification is "
        "one of the deepest results in mathematical analysis.",

        "Partial derivatives and multiple integrals extend single-variable "
        "calculus to higher dimensions. The gradient generalizes the "
        "derivative, the divergence and curl extend differential analysis "
        "to vector fields, and double and triple integrals generalize "
        "single integrals. These extensions are essential for physics "
        "and engineering applications.",
    ]


def get_questions():
    """Returns a list of test questions with reference answers and types."""
    return [
        # -- Factual questions -----------------------------------------------
        {
            "question": "What is the formal definition of a limit?",
            "reference": "A limit uses the epsilon-delta definition: for every epsilon > 0 there exists delta > 0 such that |x - c| < delta implies |f(x) - L| < epsilon.",
            "type": "factual",
        },
        {
            "question": "What does the Fundamental Theorem of Calculus state?",
            "reference": "Part 1: if F(x) = integral from a to x of f(t)dt, then F'(x) = f(x). Part 2: integral from a to b of f(x)dx = F(b) - F(a), where F is an antiderivative of f.",
            "type": "factual",
        },
        {
            "question": "What is the gradient of a scalar function?",
            "reference": "The gradient of f(x,y,z) is the vector (df/dx, df/dy, df/dz). It points in the direction of steepest ascent and its magnitude is the maximum rate of increase.",
            "type": "factual",
        },
        {
            "question": "What is the divergence of a vector field?",
            "reference": "The divergence of F = (P,Q,R) is dP/dx + dQ/dy + dR/dz. It measures the rate at which the field spreads out from a point. Zero divergence means the field is solenoidal.",
            "type": "factual",
        },
        {
            "question": "What is the curl of a vector field?",
            "reference": "The curl of F = (P,Q,R) is (dR/dy - dQ/dz, dP/dz - dR/dx, dQ/dx - dP/dy). It measures the rotation tendency. If curl(F) = 0, the field is irrotational.",
            "type": "factual",
        },

        # -- Relationship questions ------------------------------------------
        {
            "question": "How are derivatives and integrals related?",
            "reference": "The Fundamental Theorem of Calculus connects them: differentiation and integration are inverse processes. The integral of a derivative recovers the original function (up to a constant).",
            "type": "relationship",
        },
        {
            "question": "How does the Chain Rule relate to integration by substitution?",
            "reference": "Integration by substitution (u-substitution) is the reverse of the Chain Rule. If the integrand has the form f(g(x))*g'(x), substituting u=g(x) simplifies the integral.",
            "type": "relationship",
        },
        {
            "question": "What is the relationship between Green's Theorem and Stokes' Theorem?",
            "reference": "Green's Theorem is a special two-dimensional case of Stokes' Theorem. Stokes' Theorem generalizes Green's Theorem to surfaces in three dimensions.",
            "type": "relationship",
        },
        {
            "question": "How are gradient, divergence, and curl connected?",
            "reference": "The gradient maps scalars to vectors, divergence maps vectors to scalars, and curl maps vectors to vectors. Key identities: div(curl F) = 0 and curl(grad f) = 0.",
            "type": "relationship",
        },
        {
            "question": "How does the Product Rule relate to integration by parts?",
            "reference": "Integration by parts is derived from the Product Rule. The Product Rule d(uv) = u dv + v du, when integrated, gives integral u dv = uv - integral v du.",
            "type": "relationship",
        },

        # -- Reasoning questions ---------------------------------------------
        {
            "question": "Why does a Taylor series converge only within a radius of convergence?",
            "reference": "A power series converges where the terms become small enough to sum to a finite value. The radius of convergence R is determined by the ratio or root test applied to the coefficients.",
            "type": "reasoning",
        },
        {
            "question": "Why is the Jacobian needed when changing variables in multiple integrals?",
            "reference": "The Jacobian accounts for how the coordinate transformation stretches or compresses area/volume elements. Without it, the integral would not correctly measure the transformed region.",
            "type": "reasoning",
        },
        {
            "question": "Why are conservative vector fields path-independent?",
            "reference": "Because curl(F) = 0 for conservative fields, the field is the gradient of a potential. The line integral depends only on the potential values at the endpoints, not the path taken.",
            "type": "reasoning",
        },
        {
            "question": "How is the second derivative used to classify critical points?",
            "reference": "At a critical point where f'(c)=0: if f''(c) > 0 the function is concave up so it is a local minimum; if f''(c) < 0 the function is concave down so it is a local maximum.",
            "type": "reasoning",
        },

        # -- Summary questions -----------------------------------------------
        {
            "question": "What are the main integration techniques covered in calculus?",
            "reference": "The main techniques are: direct antidifferentiation, integration by substitution (u-substitution, reverse Chain Rule), integration by parts (from Product Rule), and partial fractions.",
            "type": "summary",
        },
        {
            "question": "How do the fundamental theorems of vector calculus unify?",
            "reference": "The Fundamental Theorem of Calculus, Green's Theorem, Stokes' Theorem, and the Divergence Theorem are all cases of the generalized Stokes' theorem, relating boundary integrals to interior integrals.",
            "type": "summary",
        },
        {
            "question": "What are the key applications of derivatives?",
            "reference": "Derivatives are used for: finding rates of change, optimization (maxima/minima), related rates problems, analyzing concavity and inflection points, and linear approximation via differentials.",
            "type": "summary",
        },
        {
            "question": "How does single-variable calculus extend to multiple dimensions?",
            "reference": "Derivatives extend to partial derivatives and gradients. Single integrals extend to double and triple integrals. The Chain Rule extends via total derivatives. Line and surface integrals generalize to paths and surfaces in space.",
            "type": "summary",
        },
    ]
