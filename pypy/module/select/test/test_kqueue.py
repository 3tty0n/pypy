from pypy.conftest import gettestobjspace


class AppTestKqueue(object):
    def setup_class(cls):
        import errno
        import select

        cls.space = gettestobjspace(usemodules=["select", "_socket", "posix"])

        if not hasattr(select, "kqueue"):
            py.test.skip("test requires BSD")

    def test_create(self):
        import select

        kq = select.kqueue()
        assert kq.fileno() > 0
        assert not kq.closed
        kq.close()
        assert kq.closed
        raises(ValueError, kq.fileno)

    def test_create_event(self):
        import select
        import sys

        fd = sys.stderr.fileno()
        ev = select.kevent(fd)
        other = select.kevent(1000)
        assert ev.ident == fd
        assert ev.filter == select.KQ_FILTER_READ
        assert ev.flags == select.KQ_EV_ADD
        assert ev.fflags == 0
        assert ev.data == 0
        assert ev.udata == 0
        assert ev == ev
        assert ev != other
        assert cmp(ev, other) == -1
        assert ev < other
        assert other >= ev
        raises(TypeError, cmp, ev, None)
        raises(TypeError, cmp, ev, 1)
        raises(TypeError, cmp, ev, "ev")

        ev = select.kevent(fd, select.KQ_FILTER_WRITE)
        assert ev.ident == fd
        assert ev.filter == select.KQ_FILTER_WRITE
        assert ev.flags == select.KQ_EV_ADD
        assert ev.fflags == 0
        assert ev.data == 0
        assert ev.udata == 0
        assert ev == ev
        assert ev != other

        ev = select.kevent(fd, select.KQ_FILTER_WRITE, select.KQ_EV_ONESHOT)
        assert ev.ident == fd
        assert ev.filter == select.KQ_FILTER_WRITE
        assert ev.flags == select.KQ_EV_ONESHOT
        assert ev.fflags == 0
        assert ev.data == 0
        assert ev.udata == 0
        assert ev == ev
        assert ev != other

        ev = select.kevent(1, 2, 3, 4, 5, 6)
        assert ev.ident == 1
        assert ev.filter == 2
        assert ev.flags == 3
        assert ev.fflags == 4
        assert ev.data == 5
        assert ev.udata == 6
        assert ev == ev
        assert ev != other

    def test_queue_event(self):
        import errno
        import select
        import socket
        import sys

        server_socket = socket.socket()
        server_socket.bind(("127.0.0.1", 0))
        server_socket.listen(1)
        client = socket.socket()
        client.setblocking(False)
        exc = raises(socket.error, client.connect, ("127.0.0.1", server_socket.getsockname()[1]))
        assert exc.value.args[0] == errno.EINPROGRESS
        server, addr = server_socket.accept()

        if sys.platform.startswith("darwin"):
            flags = select.KQ_EV_ADD | select.KQ_EV_ENABLE
        else:
            flags = 0

        kq1 = select.kqueue()
        kq2 = select.kqueue.fromfd(kq.fileno())

        ev = select.kevent(server.fileno(), select.KQ_FILTER_WRITE, select.KQ_EV_ADD | select.KQ_EV_ENABLE)
        kq1.control([ev], 0)
        ev = select.kevent(server.fileno(), select.KQ_FILTER_READ, select.KQ_EV_ADD | select.KQ_EV_ENABLE)
        kq1.control([ev], 0)
        ev = select.kevent(client.fileno(), select.KQ_FILTER_WRITE, select.KQ_EV_ADD | select.KQ_EV_ENABLE)
        kq2.control([ev], 0)
        ev = select.kevent(client.fileno(), select.KQ_FILTER_READ, select.KQ_EV_ADD | select.KQ_EV_ENABLE)
        kq2.control([ev], 0)

        events = kq1.control(None, 4, 1)
        events = [(e.ident, e.filter, e.flags) for e in events]
        events.sort()
        assert events == [
            (client.fileno(), select.KQ_FILTER_WRITE, flags),
            (server.fileno(), select.KQ_FILTER_WRITE, flags),
        ]
        client.send("Hello!")
        server.send("world!!!")

        for i in xrange(10):
            events = kq1.control(None, 4, 1)
            if len(events) == 4:
                break
            time.sleep(.1)
        else:
            assert False, "timeout waiting for event notification"

        events = [(e.ident, e.filter, e.flags) for e in events]
        events.sort()
        assert events == [
            (client.fileno(), select.KQ_FILTER_WRITE, flags),
            (client.fileno(), select.KQ_FILTER_READ, flags),
            (server.fileno(), select.KQ_FILTER_WRITE, flags),
            (server.fileno(), select.KQ_FILTER_READ, flags),
        ]

        ev = select.kevent(client.fileno(), select.KQ_FILTER_WRITE, select.KQ_EV_DELETE)
        kq1.control([ev], 0)
        ev = select.kevent(client.fileno(), select.KQ_FILTER_READ, select.KQ_EV_DELETE)
        kq1.control([ev], 0)
        ev = select.kevent(server.fileno(), select.KQ_FILTER_READ, select.KQ_EV_DELETE)
        kq1.control([ev], 0, 0)

        events = kq.control([], 4, 0.99)
        events = [(e.ident, e.filter, e.flags) for e in events]
        events.sort()
        assert events == [
            (serve.fileno(), select.KQ_FILTER_WRITE, flags),
        ]

        client.close()
        server.close()
        server_socket.close()

    def test_pair(self):
        import select
        import socket

        kq = select.kqueue()
        a, b = socket.socketpair()

        a.send('foo')
        event1 = select.kevent(a, select.KQ_FILTER_READ, select.KQ_EV_ADD | select.KQ_EV_ENABLE)
        event2 = select.kevent(b, select.KQ_FILTER_READ, select.KQ_EV_ADD | select.KQ_EV_ENABLE)
        r = kq.control([event1, event2], 1, 1)
        assert r
        assert r[0].flags & select.KQ_EV_ERROR
        data = b.recv(r[0].data)
        assert data == 'foo'

        a.close()
        b.close()
        kq.close()