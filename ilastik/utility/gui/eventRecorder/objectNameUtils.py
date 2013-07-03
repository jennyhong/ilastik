import time

from PyQt4.QtCore import QObject
from PyQt4.QtGui import QApplication, QWidget

def get_fully_qualified_name(obj):
    """
    Return a fully qualified object name of the form: someobject.somechild.somegrandchild.etc
    Before returning, this function **renames** any children that don't have unique names within their parent.

    Note: The name uniqueness check and renaming algorithm are terribly inefficient, 
          but it doesn't seem to slow things down much.  We could improve this later if it becomes a problem.
    """
    # Must call QObject.parent this way because obj.parent() is *shadowed* in 
    #  some subclasses (e.g. QModelIndex), which really is very ugly on Qt's part.
    parent = QObject.parent(obj)
    objName = obj.objectName()
    if objName == "":
        _assign_default_object_name(obj)
    if not _has_unique_name(obj):
        _normalize_child_names(parent)
    
    objName = str(obj.objectName())
    
    # We combine object names using periods, which means they better not have periods themselves...
    assert objName.find('.') == -1, "Objects names must not use periods!  Found an object named: {}".format( objName )

    if parent is None:
        return objName
    
    fullname = "{}.".format( get_fully_qualified_name(parent) ) + objName

    # Make sure no siblings have the same name!
    assert _has_unique_name(obj), "Detected multiple objects with full name: {}".format( fullname )

    return fullname

def get_named_object(full_name, timeout=10.0):
    """
    Locate the object with the given fully qualified name.
    While searching for the object, actively **rename** any objects that do not have unique names within their parent.
    Since the renaming scheme is consistent with get_fully_qualified name, we should always be able to locate the target object, even if it was renamed when the object was originally recorded.
    """
    timeout_ = timeout
    obj = _locate_descendent(None, full_name)
    while obj is None and timeout > 0.0:
        time.sleep(1.0)
        timeout -= 1.0
        obj = _locate_descendent(None, full_name)

    if obj is None:
        # We couldn't find the child.
        # To give a better error message, find the deepest object that COULD be found
        names = full_name.split('.')
        for i in range(len(names)):
            ancestor_name = ".".join( names[:-i-1] )
            obj = _locate_descendent(None, ancestor_name)
            if obj is not None:
                break
            else:
                ancestor_name = None

        msg = "Couldn't locate object: {} within timeout of {} seconds\n".format( full_name, timeout_ )
        if ancestor_name:
            msg += "Deepest found object was: {}".format( ancestor_name )
        else:
            msg += "Failed to find the top-level widget {}".format( full_name.split('.')[0] )
        raise RuntimeError( msg )
    return obj

def _assign_default_object_name( obj ):
    parent = QObject.parent(obj)
    if parent is None:
        # We just name the object after it's type and hope for the best.
        obj.setObjectName( obj.__class__.__name__ )
    else:
        index = parent.children().index( obj )
        newname = "child_{}_{}".format( index, obj.__class__.__name__ )
        existing_names = map( QObject.objectName, parent.children() )
        assert newname not in existing_names, "Children were not accessed in the expected order, so renaming is not consistent! Parent widget: {} already has a child with name: {}".format( get_fully_qualified_name(parent), newname )
        obj.setObjectName( newname )

def _has_unique_name(obj):
    parent = QObject.parent(obj)
    if parent is None:
        return True # We assume that top-level widgets are uniquely named
                    # Note that 'garbage' widgets may have parent=None as well.  
                    # In that case, we don't care about their names, AS LONG AS THEY AREN"T TOP-LEVEL.
    obj_name = obj.objectName()
    for child in parent.children():
        if child is not obj and child.objectName() == obj_name:
            # If the conflicting child is hidden, it doesn't count as a real conflict.
            if isinstance(child, QWidget) and child.isVisible():
                return False
    return True

def _normalize_child_names(parent):
    """
    Make sure no two children of parent have the same name.
    If two children have the same name, only rename the second one.
    """
    existing_names = set()
    for child in parent.children():
        if child.objectName() in existing_names:
            _assign_default_object_name(child)
        existing_names.add( child.objectName() )

def _locate_immediate_child(parent, childname):
    if parent is None:
        siblings = QApplication.topLevelWidgets()
    else:
        # Only consider visible children (or non-widgets)
        siblings = parent.children()
        def isVisible(obj):
            return not isinstance(obj, QWidget) or obj.isVisible()
        siblings = filter( isVisible, siblings )

    for child in siblings:
        if child.objectName() == "":
            _assign_default_object_name(child)
        if parent is not None and not _has_unique_name(child):
            _normalize_child_names(parent)
        if child.objectName() == childname:
            return child
    return None

def _locate_descendent(parent, full_name):
    names = full_name.split('.')
    assert names[0] != ''
    child = _locate_immediate_child(parent, names[0])
    if len(names) == 1:
        return child
    else:
        return _locate_descendent( child, '.'.join(names[1:]) )
