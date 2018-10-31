import sys


def check_modules(modules, non_required_modules):
    """ Verify that non-required modules are not loaded when importing others.

    The non-required modules are considered not part of garage.
    An assertion error is thrown if some of the non-required modules are
    loaded.

    Parameters:
        modules: list of strings with the modules to import.
        non_required_modules: list of modules that shouldn't be imported.
    """

    # Obtain the modules imported by python
    modules_sys = set(sys.modules)

    # Import the modules passed as parameter and obtain all modules that
    # get imported by the chain of calls.
    for module in modules:
        __import__(module)
    modules_tf = set(sys.modules)

    # Filter the list of modules
    diff = modules_tf - modules_sys
    diff = sorted(list(diff))
    diff = [modulename for modulename in diff if modulename[0] != "_"]
    diff = [modulename for modulename in diff if not modulename[0].isdigit()]

    # Print the list of garage modules that get imported
    garage_module_names = sorted(
        list(
            set([
                modulename for modulename in diff
                if modulename.startswith("garage")
            ])))
    print("Garage modules:")
    for modulename in garage_module_names:
        print(modulename)

    # Print the list of non-garage modules that get imported and identify
    # those that are not required.
    non_required_found = []
    module_names = sorted(
        list(
            set([
                modulename.split(".", 1)[0] for modulename in diff
                if not modulename.startswith("garage")
            ])))
    print("\nNon-garage modules:")
    for modulename in module_names:
        if (any([
                nonrequired == modulename
                for nonrequired in non_required_modules
        ])):
            non_required_found.append(modulename)
        print(modulename)

    assert not non_required_found, ("The following modules shouldn't be "
                                    "imported"
                                    ":\n%s" % ("\n".join(non_required_found)))
