def run():
    """
    Run the nose test scripts for QuTiP.
    """
    # Call about to get all version info printed with tests
    import nose
    # runs tests in qutip.tests module only
    nose.run(defaultTest="rb_cqed.tests", argv=['nosetests', '-v'])