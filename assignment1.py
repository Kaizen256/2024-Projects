import math

def print_cat():
    """ 
        https://www.asciiart.eu/animals/cats
        Prints ASCII art of an cat
    """
    print("  /\\_/\\  (\n ( ^'^ ) _)\n   \\\"/  (\n ( | | )\n(__d b__)")

def print_owl():
    """ 
        https://www.asciiart.eu/animals/birds-land
        Prints ASCII art of an owl 
    """
    print("  , _ ,\n ( o o )\n/'` ' `'\\\n|'''''''|\n|\\\\'''//|")

def print_logo():
    """
        Prints alternating ASCII images of a cat and an owl
    """
    print("/~~~~~~~~\\")
    print_cat()
    print("/~~~~~~~~\\")
    print_owl()
    print("/~~~~~~~~\\")
    print_cat()
    print("/~~~~~~~~\\")
    print_owl()
    print("/~~~~~~~~\\")
    
print_logo()

def calculate_surface_area(height:float,diameter:float):
    """
        Calculates and prints the surface area of a cylinder and rounds it to one decimal.
        Parameters:
            height (float): The height of the cylinder. Must be a non-negative number.
            diameter (float): The diameter of the cylinder. Must be a non-negative number.

        Examples:
            calculate_surface_area(6.5, 2.2) --> 52.5
            calculate_surface_area(-6.5, -2.2) --> ValueError
            calculate_surface_area(1.0, 0.0) --> 0.0
            calculate_surface_area(1.2, 3.5) --> 32.4
    
    """

    if height < 0 or diameter < 0:
        raise ValueError("Must be positive numbers")
        #I have a fair bit of experience in python, thats why I know Error Handling. Even though it isn't required I added it in anyways.
        
    circumference = diameter * math.pi
    top_area = (diameter/2)**2 * math.pi
    wall_area = circumference * height
    print("cylinder area: {0:.1f}".format(2*top_area + wall_area))

calculate_surface_area(1.2, 3.5)

    

