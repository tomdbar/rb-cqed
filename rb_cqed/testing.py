def run():
    """
    Run the nose test scripts for rb_cqed.
    """
    import nose
    # runs tests in rb_cqed.tests module only
    nose.run(defaultTest="rb_cqed.tests", argv=['nosetests', '-v'])