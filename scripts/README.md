The `HowToGalaxy` folder contains **HowToGalaxy** lectures, which teach a new user how to model a galaxy.

# Folders

- `chapter_1_introduction`: An introduction to galaxy morphology and structure using **PyAutoGalaxy**.
- `chapter_2_modeling`: How to model galaxies, including a primer on Bayesian non-linear analysis.
- `chapter_3_pixelizations`: How to perform pixelized reconstructions of a galaxy.
- `chapter_4_scaling_up_galaxies`: How to scale up modeling to multiple galaxies, from blended pairs to cluster fields.
- `chapter_optional`: Optional tutorials.

# Full Explanation

Welcome to **HowToGalaxy** - The **PyAutoGalaxy** tutorial!

# HOW TO TACKLE HowToGalaxy

The **HowToGalaxy** lecture series currently sits at 4 chapters, and each will take a day or so to go through
properly. You probably want to be modeling galaxies faster than that! Furthermore, the concepts
in the later chapters are pretty challenging, and familiarity and modeling is desirable before
you tackle them.

Therefore, we recommend that you complete chapters 1 & 2 and then apply what you've learnt to the modeling of simulated
and real galaxy data, using the scripts found in the 'autogalaxy_workspace'. Once you're happy
with the results and confident with your use of **PyAutoGalaxy**, you can then begin to cover the advanced functionality
covered in chapters 3 & 4.

# JUYPTER NOTEBOOKS

All tutorials are supplied as Jupyter Notebooks, which come with a '.ipynb' suffix. For those new to Python, Jupyter
Notebooks are a different way to write, view and use Python code. Compared to the traditional Python scripts,
they allow:

- Small blocks of code to be viewed and run at a time.
- Images and visualization from a code to be displayed directly underneath it.
- Text script to appear between the blocks of code.

This makes them an ideal way for us to present the **HowToGalaxy** lecture series, therefore I recommend you get yourself
a Jupyter notebook viewer (<https://jupyter.org/>) if you have not done so already.

If you *really* want to use Python scripts, all tutorials are supplied a \`\` python files in the 'scripts' folder of
each chapter.

For actual **PyAutoGalaxy** use, I recommend you use Python scripts. Therefore, as you go through the lecture series
you will notice that we will transition you to Python scripts in the third chapter.

# VISUALIZATION

Before beginning the **HowToGalaxy** lecture series, in chapter 1 you should do 'tutorial_0_visualization'. This will
take you through how **PyAutoGalaxy** interfaces with matplotlib to perform visualization and will get you setup such that
images and figures display correctly in your Jupyter notebooks.

# CODE STYLE AND FORMATTING

When you begin the notebooks, you may notice the style and formatting of our Python code looks different to what you
are used to. For example, it is common for brackets to be placed on their own line at the end of function calls,
the inputs of a function or class may be listed over many separate lines and the code in general takes up a lot more
space then you are used to.

This is intentional, because we believe it makes the cleanest, most readable code possible. In fact - lots of people do,
which is why we use an auto-formatter to produce the code in a standardized format. If you're interested in the style
and would like to adapt it to your own code, check out the Python auto-code formatter 'black'.

<https://github.com/python/black>

# OVERVIEW OF CHAPTER 1 (Beginner)

**Galaxy Structure with PyAutoGalaxy**

In chapter 1, we'll learn about galaxy structure and **PyAutoGalaxy**. At the end, you'll be able to:

1. Create uniform grid's of (x,y) Cartesian coordinates.
2. Combine these grid's with light profiles to make images.
3. Combine these light profiles to make galaxies.
4. Simulate telescope CCD imaging data of a galaxy.
5. Fit imaging data with model images generated via galaxy objects.

# OVERVIEW OF CHAPTER 2 (Beginner)

**Bayesian Inference and Non-linear Searches**

In chapter 2, we'll cover Bayesian inference and model-fitting via a non-linear search. We will use these tools to
fit CCD imaging data of a galaxy with a model. At the end, you'll understand:

1. The concept of a non-linear search and non-linear parameter space.
2. How to fit a model to galaxy CCD imaging via a non-linear search.
3. The trade-off between realism and complexity when choosing a model.
4. Why an incorrect model may be inferred and how to prevent this from happening.
5. The challenges that are involved in inferring a robust model in a computationally reasonable run-time.
6. How to chain non-linear searches together to build automated modeling pipelines with prior passing.

**Once completed, you'll be ready to model your own galaxies with PyAutoGalaxy!**

# OVERVIEW OF CHAPTER 3 (Intermediate)

**Using an inversion to perform a pixelized morphology reconstruction**

In chapter 3, we'll learn how to reconstruct the morphological features of a galaxy using a pixel-grid, ensuring
that we can fit an accurate model to galaxies with complex and irregular morphologies. You'll learn how to:

1. Pixelize a galaxy reconstruction into pixels.
2. Perform a linear inversion using this pixelization to reconstruct the galaxy's light.
3. Apply a smoothness prior on the galaxy reconstruction, called regularization.
4. Apply smoothing within a Bayesian framework to objectively quantify the reconstruction's complexity.
5. Write down the linear algebra and Bayesian evidence equations that underpin the whole framework.
6. Use these features to fit a model via non-linear searches.

# OVERVIEW OF CHAPTER 4 (Advanced)

**Scaling Up Galaxies**

In chapter 4, we'll scale galaxy modeling up beyond a single galaxy, learning how to:

1. Handle extra galaxies near the one being fitted, by scaling their light out of the fit or modeling them explicitly.
2. Model two or more blended galaxies simultaneously and understand the degeneracies this creates.
3. Model cluster fields, where the brightest cluster galaxy is fitted richly and the member population is
   composed from a catalogue.
