import math

def calculate_lambda():
    try:
    # Get user inputs
        c_v = float(input("Enter the count of v: "))
        c_n = float(input("Enter the count of n: "))
        c_vp = float(input("Enter the count of vp: "))
        c_np = float(input("Enter the count of np: "))

        # Calculate lambda using the given formula
        result = math.log2(((c_vp/c_v)*(1-(c_np/c_n)))/(c_np/c_n))

        return result

    except ValueError:
        print("Invalid input. Please enter numeric values for v, n, and P(NAP = 1|n).")
    except ZeroDivisionError:
        print("Error: Division by zero. Please make sure n is not equal to 1.")

# Call the funcƟon to calculate lambda
l=calculate_lambda()
print("Lambda is : ",l)
if l>0:
    print("PP with V")
else :
    print("PP with N")