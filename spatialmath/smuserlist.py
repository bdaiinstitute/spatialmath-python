"""
Provide list super powers for spatial math objects.
"""

from collections import UserList

class SMUserList(UserList):

    @classmethod
    def Empty(cls):
        """
        Construct an empty instance (superclass method)
        
        :return: a quaternion instance with zero values
        :rtype: Quaternion

        Example::
            
            >>> q = Quaternion.Empty()
            >>> len(q)
            0
        """
        q = cls()
        q.data = []
        return q

    @property
    def _A(self):
        # get the underlying numpy array
        if len(self.data) == 1:
            return self.data[0]
        else:
            return self.data

    @property
    def _A(self):
        # get the underlying numpy array
        if len(self.data) == 1:
            return self.data[0]
        else:
            return self.data

    # ------------------------------------------------------------------------ #

    def __getitem__(self, i):
        """
        Access value of an instance (superclass method)

        :param i: index of element to return
        :type i: int
        :return: the specific element of the pose
        :rtype: Quaternion or UnitQuaternion instance
        :raises IndexError: if the element is out of bounds

        Note that only a single index is supported, slices are not.
        
        Example::
            
            >>> q = UnitQuaternion.Rx([0, 0.3, 0.6])
            >>> len(q)
            3
            >>> q[1]
            0.988771 << 0.149438, 0.000000, 0.000000 >>
        """

        if isinstance(i, slice):
            return self.__class__([self.data[k] for k in range(i.start or 0, i.stop or len(self), i.step or 1)])
        else:
            return self.__class__(self.data[i])
        
    def __setitem__(self, i, value):
        """
        Assign a value to an instance (superclass method)
        
        :param i: index of element to assign to
        :type i: int
        :param value: the value to insert
        :type value: Quaternion or UnitQuaternion instance
        :raises ValueError: incorrect type of assigned value

        Assign the argument to an element of the object's internal list of values.
        This supports the assignement operator, for example::
            
            >>> q = Quaternion([Quaternion() for i in range(10)]) # sequence of ten identity values
            >>> len(q)
            10
            >>> q[3] = Quaternion([1,2,3,4])   # assign to position 3 in the list
        """
        if not type(self) == type(value):
            raise ValueError("can't insert different type of object")
        if len(value) > 1:
            raise ValueError("can't insert a multivalued element - must have len() == 1")
        self.data[i] = value.A

    def append(self, value):
        """
        Append a value to an instance (superclass method)
        
        :param x: the value to append
        :type x: Quaternion or UnitQuaternion instance
        :raises ValueError: incorrect type of appended object

        Appends the argument to the object's internal list of values.
    
        """
        #print('in append method')
        if not type(self) == type(value):
            raise ValueError("can't append different type of object")
        if len(value) > 1:
            raise ValueError("can't append a multivalued instance - use extend")
        super().append(value.A)
        

    def extend(self, value):
        """
        Extend sequence of values in an instance (superclass method)
        
        :param x: the value to extend
        :type x: instance of same type
        :raises ValueError: incorrect type of appended object

        Appends the argument's values to the object's internal list of values.
        """
        #print('in extend method')
        if not type(self) == type(value):
            raise ValueError("can't append different type of object")
        super().extend(value._A)

    def insert(self, i, value):
        """
        Insert a value to an instance (superclass method)

        :param i: element to insert value before
        :type i: int
        :param value: the value to insert
        :type value: instance of same type
        :raises ValueError: incorrect type of inserted value

        Inserts the argument into the object's internal list of values.
        
        Examples::
            
            >>> q = UnitQuaternion()
            >>> q.insert(0, UnitQuaternion.Rx(0.1)) # insert at position 0 in the list
            >>> len(q)
            2
        """
        if not type(self) == type(value):
            raise ValueError("can't insert different type of object")
        if len(value) > 1:
            raise ValueError("can't insert a multivalued instance - must have len() == 1")
        super().insert(i, value._A)
        
    def pop(self):
        """
        Pop value from an instance (superclass method)

        :return: the first value 
        :rtype: instance of same type
        :raises IndexError: if there are no values to pop

        Removes the first quaternion value from the instancet.
        
        Example::
            
            >>> q = UnitQuaternion.Rx([0, 0.3, 0.6])
            >>> len(q)
            3
            >>> q.pop()
            1.000000 << 0.000000, 0.000000, 0.000000 >>
            >>> len(q)
            2
        """
        return self.__class__(super().pop())