#include "common_header.h"
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctype.h>
#include <setjmp.h>
#include <signal.h>
#include <search.h>

#include "structdef.h"
#include "forwarddecl.h"
#include "preimpl.h"
#include "revdb_def.h"
#include "src/rtyper.h"
#include "src/mem.h"
#include "src-revdb/revdb_include.h"

#define RDB_SIGNATURE   "RevDB:"
#define RDB_VERSION     0x00FF0003

#define WEAKREF_AFTERWARDS_DEAD    ((char)0xf2)
#define WEAKREF_AFTERWARDS_ALIVE   ((char)0xeb)

#define ASYNC_FINALIZER_TRIGGER    ((int16_t)0xff46)

#define FID_REGULAR_MODE           'R'
#define FID_SAVED_STATE            'S'
#define FID_JMPBUF_PROTECTED       'J'


typedef struct {
    Signed version;
    uint64_t reserved1, reserved2;
    void *ptr1, *ptr2;
    int reversed3;
    int argc;
    char **argv;
} rdb_header_t;


rpy_revdb_t rpy_revdb;
static char rpy_rev_buffer[16384];    /* max. 32768 */
int rpy_rev_fileno = -1;
static char flag_io_disabled = FID_REGULAR_MODE;


static void setup_record_mode(int argc, char *argv[]);
static void setup_replay_mode(int *argc_p, char **argv_p[]);
static void check_at_end(uint64_t stop_points);

RPY_EXTERN
void rpy_reverse_db_setup(int *argc_p, char **argv_p[])
{
    /* init-time setup */

    int replay_asked = (*argc_p >= 2 && !strcmp((*argv_p)[1],"--revdb-replay"));

#ifdef RPY_RDB_DYNAMIC_REPLAY
    RPY_RDB_REPLAY = replay_asked;
#else
    if (replay_asked != RPY_RDB_REPLAY) {
        fprintf(stderr, "This executable was only compiled for %s mode.\n",
                RPY_RDB_REPLAY ? "replay" : "record");
        exit(1);
    }
#endif

    if (RPY_RDB_REPLAY)
        setup_replay_mode(argc_p, argv_p);
    else
        setup_record_mode(*argc_p, *argv_p);
}

RPY_EXTERN
void rpy_reverse_db_teardown(void)
{
    uint64_t stop_points;
    if (RPY_RDB_REPLAY) {
        /* hack: prevents RPY_REVDB_EMIT() from calling
           rpy_reverse_db_fetch(), which has nothing more to fetch now */
        rpy_revdb.buf_limit += 1;
    }
    RPY_REVDB_EMIT(stop_points = rpy_revdb.stop_point_seen; ,
                   uint64_t _e, stop_points);

    if (!RPY_RDB_REPLAY) {
        rpy_reverse_db_flush();
        if (rpy_rev_fileno >= 0) {
            close(rpy_rev_fileno);
            rpy_rev_fileno = -1;
        }
    }
    else
        check_at_end(stop_points);
}

static void record_stop_point(void);
static void replay_stop_point(void);

RPY_EXTERN
void rpy_reverse_db_stop_point(void)
{
    if (!RPY_RDB_REPLAY)
        record_stop_point();
    else
        replay_stop_point();
}


/* ------------------------------------------------------------ */
/* Recording mode                                               */
/* ------------------------------------------------------------ */


static void write_all(const void *buf, ssize_t count)
{
    while (count > 0) {
        ssize_t wsize = write(rpy_rev_fileno, buf, count);
        if (wsize <= 0) {
            if (wsize == 0)
                fprintf(stderr, "writing to RevDB file: "
                                "unexpected non-blocking mode\n");
            else
                fprintf(stderr, "Fatal error: writing to RevDB file: %m\n");
            abort();
        }
        buf += wsize;
        count -= wsize;
    }
}

static void close_revdb_fileno_in_fork_child(void)
{
    if (rpy_rev_fileno >= 0) {
        close(rpy_rev_fileno);
        rpy_rev_fileno = -1;
    }
}

static void setup_record_mode(int argc, char *argv[])
{
    char *filename = getenv("PYPYRDB");
    rdb_header_t h;
    int i;

    assert(RPY_RDB_REPLAY == 0);

    if (filename && *filename) {
        putenv("PYPYRDB=");
        rpy_rev_fileno = open(filename, O_RDWR | O_CLOEXEC |
                              O_CREAT | O_NOCTTY | O_TRUNC, 0600);
        if (rpy_rev_fileno < 0) {
            fprintf(stderr, "Fatal error: can't create PYPYRDB file '%s'\n",
                    filename);
            abort();
        }
        atexit(rpy_reverse_db_flush);

        write_all(RDB_SIGNATURE, strlen(RDB_SIGNATURE));
        for (i = 0; i < argc; i++) {
            write_all("\t", 1);
            write_all(argv[i], strlen(argv[i]));
        }
        write_all("\n\0", 2);

        memset(&h, 0, sizeof(h));
        h.version = RDB_VERSION;
        h.ptr1 = &rpy_reverse_db_stop_point;
        h.ptr2 = &rpy_revdb;
        h.argc = argc;
        h.argv = argv;
        write_all((const char *)&h, sizeof(h));

        /* write the whole content of rpy_rdb_struct */
        /*write_all((const char *)&rpy_rdb_struct, sizeof(rpy_rdb_struct));*/

        fprintf(stderr, "PID %d: recording revdb log to '%s'\n",
                        (int)getpid(), filename);
    }
    else {
        fprintf(stderr, "PID %d starting, log file disabled "
                        "(use PYPYRDB=logfile)\n", (int)getpid());
    }

    rpy_revdb.buf_p = rpy_rev_buffer + sizeof(int16_t);
    rpy_revdb.buf_limit = rpy_rev_buffer + sizeof(rpy_rev_buffer) - 32;
    rpy_revdb.unique_id_seen = 1;

    pthread_atfork(NULL, NULL, close_revdb_fileno_in_fork_child);
}

static void flush_buffer(void)
{
    /* write the current buffer content to the OS */
    ssize_t full_size = rpy_revdb.buf_p - rpy_rev_buffer;
    rpy_revdb.buf_p = rpy_rev_buffer + sizeof(int16_t);
    if (rpy_rev_fileno >= 0)
        write_all(rpy_rev_buffer, full_size);
}

static ssize_t current_packet_size(void)
{
    return rpy_revdb.buf_p - (rpy_rev_buffer + sizeof(int16_t));
}

RPY_EXTERN
void rpy_reverse_db_flush(void)
{
    ssize_t content_size = current_packet_size();
    if (content_size != 0) {
        char *p = rpy_rev_buffer;
        assert(0 < content_size && content_size <= 32767);
        *(int16_t *)p = content_size;
        flush_buffer();
    }
}

void boehm_gc_finalizer_notifier(void)
{
    /* This is called by Boehm when there are pending finalizers.
       They are only invoked when we call GC_invoke_finalizers(),
       which we only do at stop points in the case of revdb. 
    */
    assert(!RPY_RDB_REPLAY);
    assert(rpy_revdb.stop_point_break <= rpy_revdb.stop_point_seen + 1);
    rpy_revdb.stop_point_break = rpy_revdb.stop_point_seen + 1;
}

static void fq_trigger(void)
{
    int i = 0;
    while (boehm_fq_trigger[i])
        boehm_fq_trigger[i++]();
}

static long in_invoke_finalizers;

static void record_stop_point(void)
{
    /* ===== FINALIZERS =====

       When the GC wants to invoke some finalizers, it causes this to
       be called at the stop point.  (This is not called at *every*
       stop point.)  The new-style finalizers are only enqueued at
       this point.  The old-style finalizers run immediately,
       conceptually just *after* the stop point.
     */
    int i;
    char *p = rpy_rev_buffer;
    int64_t done;

    /* Write an ASYNC_FINALIZER_TRIGGER packet */
    rpy_reverse_db_flush();
    assert(current_packet_size() == 0);

    *(int16_t *)p = ASYNC_FINALIZER_TRIGGER;
    memcpy(rpy_revdb.buf_p, &rpy_revdb.stop_point_seen, sizeof(uint64_t));
    rpy_revdb.buf_p += sizeof(uint64_t);
    flush_buffer();

    /* Invoke all Boehm finalizers.  For new-style finalizers, this
       will only cause them to move to the queues, where
       boehm_fq_next_dead() will be able to fetch them later.  For
       old-style finalizers, this will really call the finalizers,
       which first emit to the rdb log the uid of the object.  So
       after we do that any number of times, we emit the uid -1 to
       mean "now done, continue with the rest of the program".
    */
    in_invoke_finalizers++;
    GC_invoke_finalizers();
    in_invoke_finalizers--;
    RPY_REVDB_EMIT(done = -1;, int64_t _e, done);

    /* Now we're back in normal mode.  We trigger the finalizer 
       queues here. */
    fq_trigger();
}

RPY_EXTERN
void rpy_reverse_db_call_destructor(void *obj)
{
    /* old-style finalizers.  Should occur only from the 
       GC_invoke_finalizers() call above. 
    */
    int64_t uid;

    if (RPY_RDB_REPLAY)
        return;
    if (!in_invoke_finalizers) {
        fprintf(stderr, "call_destructor: called at an unexpected time\n");
        exit(1);
    }
    RPY_REVDB_EMIT(uid = ((struct pypy_header0 *)obj)->h_uid;, int64_t _e, uid);
}

RPY_EXTERN
Signed rpy_reverse_db_identityhash(struct pypy_header0 *obj)
{
    /* Boehm only */
    if (obj->h_hash == 0) {
        /* We never need to record anything: if h_hash is zero (which
           is the case for all newly allocated objects), then we just
           copy h_uid.  This gives a stable answer.  This would give
           0 for all prebuilt objects, but these should not have a
           null h_hash anyway.
        */
        obj->h_hash = obj->h_uid;
    }
    return obj->h_hash;
}

static uint64_t recording_offset(void)
{
    off_t base_offset;
    ssize_t extra_size = rpy_revdb.buf_p - rpy_rev_buffer;

    if (rpy_rev_fileno < 0)
        return 1;
    base_offset = lseek(rpy_rev_fileno, 0, SEEK_CUR);
    if (base_offset < 0) {
        perror("lseek");
        exit(1);
    }
    return base_offset + extra_size;
}

static void patch_prev_offset(int64_t offset, char old, char new)
{
    off_t base_offset;
    if (rpy_rev_fileno < 0)
        return;
    base_offset = lseek(rpy_rev_fileno, 0, SEEK_CUR);
    if (base_offset < 0) {
        perror("lseek");
        exit(1);
    }
    if (offset < base_offset) {
        char got;
        if (pread(rpy_rev_fileno, &got, 1, offset) != 1) {
            fprintf(stderr, "can't read log position %lld for checking: %m\n",
                    (long long)offset);
            exit(1);
        }
        if (got != old) {
            fprintf(stderr,
                    "bad byte at log position %lld (%d instead of %d)\n",
                    (long long)offset, got, old);
            exit(1);
        }
        if (pwrite(rpy_rev_fileno, &new, 1, offset) != 1) {
            fprintf(stderr, "can't patch log position %lld\n",
                    (long long)offset);
            exit(1);
        }
    }
    else {
        ssize_t buffered_size = rpy_revdb.buf_p - rpy_rev_buffer;
        int64_t buf_ofs = offset - base_offset;
        if (buf_ofs >= buffered_size) {
            fprintf(stderr, "invalid patch position %lld\n",
                    (long long)offset);
            exit(1);
        }
        if (rpy_rev_buffer[buf_ofs] != old) {
            fprintf(stderr,
                    "bad byte at log position %lld (%d instead of %d)\n",
                    (long long)offset, rpy_rev_buffer[buf_ofs], old);
            exit(1);
        }
        rpy_rev_buffer[buf_ofs] = new;
    }
}

/* keep in sync with 'REVDB_WEAKLINK' in rpython.memory.gctransform.boehm */
struct WEAKLINK {
    void *re_addr;
    long long re_off_prev;
};

RPY_EXTERN
void *rpy_reverse_db_weakref_create(void *target)
{
    /* see comments in ../test/test_weak.py */
    struct WEAKLINK *r;
    if (!RPY_RDB_REPLAY)
        r = GC_MALLOC_ATOMIC(sizeof(struct WEAKLINK));
    else
        r = GC_MALLOC(sizeof(struct WEAKLINK));

    if (!r) {
        fprintf(stderr, "out of memory for a weakref\n");
        exit(1);
    }
    r->re_addr = target;
    r->re_off_prev = 0;

    if (flag_io_disabled == FID_REGULAR_MODE) {
        char alive;
        /* Emit WEAKREF_AFTERWARDS_DEAD, but remember where we emit it.
           If we deref the weakref and it is still alive, we will patch
           it with WEAKREF_AFTERWARDS_ALIVE. */
        if (!RPY_RDB_REPLAY)
            r->re_off_prev = recording_offset();
        else
            r->re_off_prev = 1;    /* any number > 0 */

        RPY_REVDB_EMIT(alive = WEAKREF_AFTERWARDS_DEAD;, char _e, alive);

        if (!RPY_RDB_REPLAY) {
            OP_BOEHM_DISAPPEARING_LINK(&r->re_addr, target, /*nothing*/);
        }
        else {
            /* replaying: we don't make the weakref actually weak at all,
               but instead we always know if we're going to need the 
               weakref value later or not */
            switch (alive) {
            case WEAKREF_AFTERWARDS_DEAD:
                r->re_addr = NULL;
                break;
            case WEAKREF_AFTERWARDS_ALIVE:
                break;
            default:
                fprintf(stderr, "bad weakref_create byte in log\n");
                exit(1);
            }
        }
    }
    return r;
}

RPY_EXTERN
void *rpy_reverse_db_weakref_deref(void *weakref)
{
    struct WEAKLINK *r = (struct WEAKLINK *)weakref;
    void *result = r->re_addr;
    if (result && flag_io_disabled == FID_REGULAR_MODE) {
        if (r->re_off_prev < 0) {
            fprintf(stderr, "bug in weakrefs: bad previous offset %lld\n",
                    (long long)r->re_off_prev);
            exit(1);
        }
        if (r->re_off_prev == 0) {
            /* A prebuilt weakref.  Don't record anything */
        }
        else {
            char alive;
            if (!RPY_RDB_REPLAY) {
                patch_prev_offset(r->re_off_prev, WEAKREF_AFTERWARDS_DEAD,
                                                  WEAKREF_AFTERWARDS_ALIVE);
                r->re_off_prev = recording_offset();
            }
            RPY_REVDB_EMIT(alive = WEAKREF_AFTERWARDS_DEAD;, char _e, alive);

            if (RPY_RDB_REPLAY) {
                switch (alive) {
                case WEAKREF_AFTERWARDS_DEAD:
                    r->re_addr = NULL;
                    break;
                case WEAKREF_AFTERWARDS_ALIVE:
                    break;
                default:
                    fprintf(stderr, "bad weakref_deref byte in log\n");
                    exit(1);
                }
            }
        }
    }
    return result;
}

RPY_EXTERN
void rpy_reverse_db_callback_loc(int locnum)
{
    locnum += 300;
    assert(locnum < 0xFC00);
    if (!RPY_RDB_REPLAY) {
        _RPY_REVDB_EMIT_RECORD(unsigned char _e, (locnum >> 8));
        _RPY_REVDB_EMIT_RECORD(unsigned char _e, (locnum & 0xFF));
    }
}


/* ------------------------------------------------------------ */
/* Replaying mode                                               */
/* ------------------------------------------------------------ */


/* How it works: we run the same executable with different flags to
   run it in "replay" mode.  In this mode, it reads commands from
   stdin (in binary format) and writes the results to stdout.
   Notably, there is a command to ask it to fork, passing a new pair
   of pipes to the forked copy as its new stdin/stdout.  This is how
   we implement the illusion of going backward: we throw away the
   current fork, start from an earlier fork, make a new fork again,
   and go forward by the correct number of steps.  This is all
   controlled by a pure Python wrapper that is roughly generic
   (i.e. able to act as a debugger for any language).
*/

#include "src-revdb/fd_recv.c"

#define INIT_VERSION_NUMBER   0xd80100

#define CMD_FORK      (-1)
#define CMD_QUIT      (-2)
#define CMD_FORWARD   (-3)
#define CMD_FUTUREIDS (-4)
#define CMD_PING      (-5)

#define ANSWER_INIT       (-20)
#define ANSWER_READY      (-21)
#define ANSWER_FORKED     (-22)
#define ANSWER_AT_END     (-23)
#define ANSWER_BREAKPOINT (-24)

#define RECORD_BKPT_NUM   50

typedef void (*rpy_revdb_command_fn)(rpy_revdb_command_t *, RPyString *);

static int rpy_rev_sockfd;
static const char *rpy_rev_filename;
static uint64_t interactive_break = 1, finalizer_break = -1, uid_break = -1;
static uint64_t total_stop_points;
static jmp_buf jmp_buf_cancel_execution;
static void (*pending_after_forward)(void);
static RPyString *empty_string;
static uint64_t last_recorded_breakpoint_loc;
static int n_last_recorded_breakpoints;
static int last_recorded_breakpoint_nums[RECORD_BKPT_NUM];
static char breakpoint_mode = 'i';
static uint64_t *future_ids, *future_next_id;
static void *finalizer_tree, *destructor_tree;

RPY_EXTERN
void attach_gdb(void)
{
    char cmdline[80];
    sprintf(cmdline, "term -c \"gdb --pid=%d\"", getpid());
    system(cmdline);
    sleep(1);
}

static ssize_t read_at_least(void *buf, ssize_t count_min, ssize_t count_max)
{
    ssize_t result = 0;
    assert(count_min <= count_max);
    while (result < count_min) {
        ssize_t rsize = read(rpy_rev_fileno, buf + result, count_max - result);
        if (rsize <= 0) {
            if (rsize == 0)
                fprintf(stderr, "RevDB file appears truncated\n");
            else
                fprintf(stderr, "RevDB file read error: %m\n");
            exit(1);
        }
        result += rsize;
    }
    return result;
}

static void read_all(void *buf, ssize_t count)
{
    (void)read_at_least(buf, count, count);
}

static void read_sock(void *buf, ssize_t count)
{
    while (count > 0) {
        ssize_t got = read(rpy_rev_sockfd, buf, count);
        if (got <= 0) {
            fprintf(stderr, "subprocess: cannot read from control socket\n");
            exit(1);
        }
        count -= got;
        buf += got;
    }
}

static void write_sock(const void *buf, ssize_t count)
{
    while (count > 0) {
        ssize_t wrote = write(rpy_rev_sockfd, buf, count);
        if (wrote <= 0) {
            fprintf(stderr, "subprocess: cannot write to control socket\n");
            exit(1);
        }
        count -= wrote;
        buf += wrote;
    }
}

static void write_answer(int cmd, int64_t arg1, int64_t arg2, int64_t arg3)
{
    rpy_revdb_command_t c;
    memset(&c, 0, sizeof(c));
    c.cmd = cmd;
    c.arg1 = arg1;
    c.arg2 = arg2;
    c.arg3 = arg3;
    write_sock(&c, sizeof(c));
}

static RPyString *make_rpy_string(size_t length)
{
    RPyString *s = malloc(sizeof(RPyString) + length);
    if (s == NULL) {
        fprintf(stderr, "out of memory for a string of %llu chars\n",
                (unsigned long long)length);
        exit(1);
    }
    /* xxx assumes Boehm here for now */
    memset(s, 0, sizeof(RPyString));
    RPyString_Size(s) = length;
    return s;
}

static void reopen_revdb_file(const char *filename)
{
    rpy_rev_fileno = open(filename, O_RDONLY | O_NOCTTY);
    if (rpy_rev_fileno < 0) {
        fprintf(stderr, "Can't open file '%s': %m\n", filename);
        exit(1);
    }
}

static void set_revdb_breakpoints(void)
{
    /* note: these are uint64_t, so '-1' is bigger than positive values */
    rpy_revdb.stop_point_break = (interactive_break < finalizer_break ?
                                  interactive_break : finalizer_break);
    rpy_revdb.unique_id_break = uid_break;
    rpy_revdb.watch_enabled = (breakpoint_mode != 'i');
}

static void setup_replay_mode(int *argc_p, char **argv_p[])
{
    int argc = *argc_p;
    char **argv = *argv_p;
    rdb_header_t h;
    char input[16];
    ssize_t count;

    if (argc != 4) {
        fprintf(stderr, "syntax: %s --revdb-replay <RevDB-file> <socket_fd>\n",
                argv[0]);
        exit(2);
    }
    rpy_rev_filename = argv[2];
    reopen_revdb_file(rpy_rev_filename);
    rpy_rev_sockfd = atoi(argv[3]);

    assert(RPY_RDB_REPLAY == 1);

    read_all(input, strlen(RDB_SIGNATURE));
    if (strncmp(input, RDB_SIGNATURE, strlen(RDB_SIGNATURE)) != 0) {
        fprintf(stderr, "'%s' is not a RevDB file (or wrong platform)\n",
                rpy_rev_filename);
        exit(1);
    }
    fprintf(stderr, "%s", RDB_SIGNATURE);
    while ((read_all(input, 1), input[0] != 0))
        fputc(input[0] == '\t' ? ' ' : input[0], stderr);

    read_all(&h, sizeof(h));
    if (h.version != RDB_VERSION) {
        fprintf(stderr, "RevDB file version mismatch (got %lx, expected %lx)\n",
                (long)h.version, (long)RDB_VERSION);
        exit(1);
    }
    if (h.ptr1 != &rpy_reverse_db_stop_point ||
        h.ptr2 != &rpy_revdb) {
        fprintf(stderr,
                "\n"
                "In the replaying process, the addresses are different than\n"
                "in the recording process.  We don't support this case for\n"
                "now, sorry.  On Linux, check if Address Space Layout\n"
                "Randomization (ASLR) is enabled, and disable it with:\n"
                "\n"
                "    echo 0 | sudo tee /proc/sys/kernel/randomize_va_space\n"
                "\n");
        exit(1);
    }
    *argc_p = h.argc;
    *argv_p = h.argv;

    count = lseek(rpy_rev_fileno, 0, SEEK_CUR);
    if (count < 0 ||
            lseek(rpy_rev_fileno, -sizeof(uint64_t), SEEK_END) < 0 ||
            read(rpy_rev_fileno, &total_stop_points,
                 sizeof(uint64_t)) != sizeof(uint64_t) ||
            lseek(rpy_rev_fileno, count, SEEK_SET) != count) {
        fprintf(stderr, "%s: %m\n", rpy_rev_filename);
        exit(1);
    }

    /* read the whole content of rpy_rdb_struct */
    /*read_all((char *)&rpy_rdb_struct, sizeof(rpy_rdb_struct));*/

    rpy_revdb.buf_p = rpy_rev_buffer;
    rpy_revdb.buf_limit = rpy_rev_buffer;
    rpy_revdb.buf_readend = rpy_rev_buffer;
    rpy_revdb.stop_point_seen = 0;
    rpy_revdb.unique_id_seen = 1;
    set_revdb_breakpoints();

    empty_string = make_rpy_string(0);

    write_answer(ANSWER_INIT, INIT_VERSION_NUMBER, total_stop_points, 0);

    /* ignore the SIGCHLD signals so that child processes don't become
       zombies */
    signal(SIGCHLD, SIG_IGN);

    /* initiate the read, which is always at least one byte ahead of
       RPY_REVDB_EMIT() in order to detect the ASYNC_* operations
       early enough. */
    rpy_reverse_db_fetch(__FILE__, __LINE__);
}

static void fetch_more(ssize_t keep, ssize_t expected_size)
{
    ssize_t rsize;
    if (rpy_revdb.buf_p != rpy_rev_buffer)
        memmove(rpy_rev_buffer, rpy_revdb.buf_p, keep);
    rsize = read_at_least(rpy_rev_buffer + keep,
                          expected_size - keep,
                          sizeof(rpy_rev_buffer) - keep);
    rpy_revdb.buf_p = rpy_rev_buffer;
    rpy_revdb.buf_readend = rpy_rev_buffer + keep + rsize;
    /* rpy_revdb.buf_limit is not set */
}

RPY_EXTERN
void rpy_reverse_db_fetch(const char *file, int line)
{
    if (flag_io_disabled == FID_REGULAR_MODE) {
        ssize_t keep;
        ssize_t full_packet_size;
        int16_t header;

        if (finalizer_break != (uint64_t)-1) {
            fprintf(stderr, "reverse_db_fetch: finalizer_break != -1\n");
            exit(1);
        }
        if (rpy_revdb.buf_limit != rpy_revdb.buf_p) {
            fprintf(stderr, "bad log format: incomplete packet\n");
            exit(1);
        }

        keep = rpy_revdb.buf_readend - rpy_revdb.buf_p;
        assert(keep >= 0);

        if (keep < sizeof(int16_t)) {
            /* 'keep' does not even contain the next packet header */
            fetch_more(keep, sizeof(int16_t));
            keep = rpy_revdb.buf_readend - rpy_rev_buffer;
        }
        header = *(int16_t *)rpy_revdb.buf_p;
        if (header < 0) {
            int64_t bp;

            switch (header) {

            case ASYNC_FINALIZER_TRIGGER:
                //fprintf(stderr, "ASYNC_FINALIZER_TRIGGER\n");
                if (finalizer_break != (uint64_t)-1) {
                    fprintf(stderr, "unexpected multiple "
                                    "ASYNC_FINALIZER_TRIGGER\n");
                    exit(1);
                }
                full_packet_size = sizeof(int16_t) + sizeof(int64_t);
                if (keep < full_packet_size)
                    fetch_more(keep, full_packet_size);
                memcpy(&bp, rpy_revdb.buf_p + sizeof(int16_t), sizeof(int64_t));
                rpy_revdb.buf_p += full_packet_size;
                if (bp <= rpy_revdb.stop_point_seen) {
                    fprintf(stderr, "invalid finalizer break point\n");
                    exit(1);
                }
                finalizer_break = bp;
                set_revdb_breakpoints();
                /* Now we should not fetch anything more until we reach
                   that finalizer_break point. */
                rpy_revdb.buf_limit = rpy_revdb.buf_p;
                return;

            default:
                fprintf(stderr, "bad packet header %d\n", (int)header);
                exit(1);
            }
        }
        full_packet_size = sizeof(int16_t) + header;
        if (keep < full_packet_size)
            fetch_more(keep, full_packet_size);
        rpy_revdb.buf_limit = rpy_revdb.buf_p + full_packet_size;
        rpy_revdb.buf_p += sizeof(int16_t);
    }
    else {
        /* this is called when we are in execute_rpy_command(): we are
           running some custom code now, and we can't just perform I/O
           or access raw memory---because there is no raw memory! 
        */
        fprintf(stderr, "%s:%d: Attempted to do I/O or access raw memory\n",
                file, line);
        if (flag_io_disabled == FID_JMPBUF_PROTECTED) {
            longjmp(jmp_buf_cancel_execution, 1);
        }
        else {
            fprintf(stderr, "but we are not in a jmpbuf_protected section\n");
            exit(1);
        }
    }
}

static rpy_revdb_t saved_state;
static void *saved_exc[2];

static void change_flag_io_disabled(char oldval, char newval)
{
    if (flag_io_disabled != oldval) {
        fprintf(stderr, "change_flag_io_disabled(%c, %c) but got %c\n",
                oldval, newval, flag_io_disabled);
        exit(1);
    }
    flag_io_disabled = newval;
}

static void save_state(void)
{
    /* The program is switching from replaying execution to 
       time-paused mode.  In time-paused mode, we can run more
       app-level code like watch points or interactive prints,
       but they must not be matched against the log, and they must
       not involve generic I/O.
    */
    change_flag_io_disabled(FID_REGULAR_MODE, FID_SAVED_STATE);

    saved_state = rpy_revdb;   /* save the complete struct */

    rpy_revdb.unique_id_seen = (-1ULL) << 63;
    rpy_revdb.watch_enabled = 0;
    rpy_revdb.stop_point_break = -1;
    rpy_revdb.unique_id_break = -1;
    rpy_revdb.buf_p = rpy_rev_buffer;       /* anything readable */
    rpy_revdb.buf_limit = rpy_rev_buffer;   /* same as buf_p */
}

static void restore_state(void)
{
    /* The program is switching from time-paused mode to replaying
       execution. */
    change_flag_io_disabled(FID_SAVED_STATE, FID_REGULAR_MODE);

    /* restore the complete struct */
    rpy_revdb = saved_state;

    /* set the breakpoint fields to the current value of the *_break
       global variables, which may be different from what is in
       'save_state' */
    set_revdb_breakpoints();
}

static void protect_jmpbuf(void)
{
    change_flag_io_disabled(FID_SAVED_STATE, FID_JMPBUF_PROTECTED);
    saved_exc[0] = pypy_g_ExcData.ed_exc_type;
    saved_exc[1] = pypy_g_ExcData.ed_exc_value;
    pypy_g_ExcData.ed_exc_type = NULL;
    pypy_g_ExcData.ed_exc_value = NULL;
}

static void unprotect_jmpbuf(void)
{
    change_flag_io_disabled(FID_JMPBUF_PROTECTED, FID_SAVED_STATE);
    if (pypy_g_ExcData.ed_exc_type != NULL) {
        fprintf(stderr, "Command crashed with %.*s\n",
                (int)(pypy_g_ExcData.ed_exc_type->ov_name->rs_chars.length),
                pypy_g_ExcData.ed_exc_type->ov_name->rs_chars.items);
        exit(1);
    }
    pypy_g_ExcData.ed_exc_type = saved_exc[0];
    pypy_g_ExcData.ed_exc_value = saved_exc[1];
}

static void execute_rpy_function(rpy_revdb_command_fn func,
                                 rpy_revdb_command_t *cmd,
                                 RPyString *extra)
{
    protect_jmpbuf();
    if (setjmp(jmp_buf_cancel_execution) == 0)
        func(cmd, extra);
    unprotect_jmpbuf();
}

static void check_at_end(uint64_t stop_points)
{
    char dummy[1];
    if (rpy_revdb.buf_p != rpy_revdb.buf_limit - 1 ||
            read(rpy_rev_fileno, dummy, 1) > 0) {
        fprintf(stderr, "RevDB file error: too much data: corrupted file, "
                        "revdb bug, or non-deterministic run, e.g. a "
                        "watchpoint with side effects)\n");
        exit(1);
    }
    if (stop_points != rpy_revdb.stop_point_seen) {
        fprintf(stderr, "Bad number of stop points "
                "(seen %lld, recorded %lld)\n",
                (long long)rpy_revdb.stop_point_seen,
                (long long)stop_points);
        exit(1);
    }
    if (stop_points != total_stop_points) {
        fprintf(stderr, "RevDB file modified while reading?\n");
        exit(1);
    }

    write_answer(ANSWER_AT_END, 0, 0, 0);
    exit(0);
}

static void command_fork(void)
{
    int child_sockfd;
    int child_pid;
    off_t rev_offset = lseek(rpy_rev_fileno, 0, SEEK_CUR);

    if (ancil_recv_fd(rpy_rev_sockfd, &child_sockfd) < 0) {
        fprintf(stderr, "cannot fetch child control socket: %m\n");
        exit(1);
    }
    child_pid = fork();
    if (child_pid == -1) {
        perror("fork");
        exit(1);
    }
    if (child_pid == 0) {
        /* in the child */
        if (close(rpy_rev_sockfd) < 0) {
            perror("close");
            exit(1);
        }
        rpy_rev_sockfd = child_sockfd;

        /* Close and re-open the revdb log file in the child process.
           This is the simplest way I found to give 'rpy_rev_fileno'
           its own offset, independent from the parent.  It assumes
           that the revdb log file is still the same.  So for Linux,
           we try to open "/proc/self/fd/%d" instead. */
        char fd_filename[48];
        struct stat st;
        const char *filename;
        int old_fd = rpy_rev_fileno;

        sprintf(fd_filename, "/proc/self/fd/%d", old_fd);
        if (lstat(fd_filename, &st) == 0)
            filename = fd_filename;
        else
            filename = rpy_rev_filename;
        reopen_revdb_file(filename);

        if (close(old_fd) < 0) {
            perror("close");
            exit(1);
        }
        if (lseek(rpy_rev_fileno, rev_offset, SEEK_SET) < 0) {
            perror("lseek");
            exit(1);
        }
    }
    else {
        /* in the parent */
        write_answer(ANSWER_FORKED, child_pid, 0, 0);
        close(child_sockfd);
    }
}

static void answer_recorded_breakpoint(void)
{
    int i;
    for (i = 0; i < n_last_recorded_breakpoints; i++)
        write_answer(ANSWER_BREAKPOINT, last_recorded_breakpoint_loc,
                     0, last_recorded_breakpoint_nums[i]);
    n_last_recorded_breakpoints = 0;
}

static void command_forward(rpy_revdb_command_t *cmd)
{
    if (cmd->arg1 < 0) {
        fprintf(stderr, "CMD_FORWARD: negative step\n");
        exit(1);
    }
    assert(flag_io_disabled == FID_SAVED_STATE);
    interactive_break = saved_state.stop_point_seen + cmd->arg1;
    breakpoint_mode = (char)cmd->arg2;
    if (breakpoint_mode == 'r') {
        n_last_recorded_breakpoints = 0;
        pending_after_forward = &answer_recorded_breakpoint;
    }
}

static void command_future_ids(rpy_revdb_command_t *cmd, char *extra)
{
    free(future_ids);
    if (cmd->extra_size == 0) {
        future_ids = NULL;
        uid_break = 0;
    }
    else {
        assert(cmd->extra_size % sizeof(uint64_t) == 0);
        future_ids = malloc(cmd->extra_size + sizeof(uint64_t));
        if (future_ids == NULL) {
            fprintf(stderr, "out of memory for a buffer of %llu chars\n",
                    (unsigned long long)cmd->extra_size);
            exit(1);
        }
        memcpy(future_ids, extra, cmd->extra_size);
        future_ids[cmd->extra_size / sizeof(uint64_t)] = 0;
        uid_break = *future_ids;
        //attach_gdb();
    }
    future_next_id = future_ids;
}

static void command_default(rpy_revdb_command_t *cmd, char *extra)
{
    RPyString *s;
    int i;
    for (i = 0; rpy_revdb_commands.rp_names[i] != cmd->cmd; i++) {
        if (rpy_revdb_commands.rp_names[i] == 0) {
            fprintf(stderr, "unknown command %d\n", cmd->cmd);
            exit(1);
        }
    }

    if (cmd->extra_size == 0) {
        s = empty_string;
    }
    else {
        s = make_rpy_string(cmd->extra_size);
        memcpy(_RPyString_AsString(s), extra, cmd->extra_size);
    }
    execute_rpy_function(rpy_revdb_commands.rp_funcs[i], cmd, s);
}

RPY_EXTERN
void rpy_reverse_db_watch_save_state(void)
{
    save_state();
}

RPY_EXTERN
void rpy_reverse_db_watch_restore_state(bool_t any_watch_point)
{
    restore_state();
    rpy_revdb.watch_enabled = any_watch_point;
}

static void replay_call_destructors(void);

static void replay_stop_point(void)
{
    if (finalizer_break != (uint64_t)-1)
        replay_call_destructors();

    if (rpy_revdb.stop_point_break != interactive_break) {
        fprintf(stderr, "mismatch between interactive_break and "
                        "stop_point_break\n");
        exit(1);
    }

    while (rpy_revdb.stop_point_break == rpy_revdb.stop_point_seen) {
        save_state();

        if (pending_after_forward) {
            void (*fn)(void) = pending_after_forward;
            pending_after_forward = NULL;
            fn();
        }
        else {
            rpy_revdb_command_t cmd;
            write_answer(ANSWER_READY,
                         saved_state.stop_point_seen,
                         saved_state.unique_id_seen,
                         0);
            read_sock(&cmd, sizeof(cmd));

            char extra[cmd.extra_size + 1];
            extra[cmd.extra_size] = 0;
            if (cmd.extra_size > 0)
                read_sock(extra, cmd.extra_size);

            switch (cmd.cmd) {

            case CMD_FORK:
                command_fork();
                break;

            case CMD_QUIT:
                exit(0);
                break;

            case CMD_FORWARD:
                command_forward(&cmd);
                break;

            case CMD_FUTUREIDS:
                command_future_ids(&cmd, extra);
                break;

            case CMD_PING:     /* to get only the ANSWER_READY */
                break;

            default:
                command_default(&cmd, extra);
                break;
            }
        }
        restore_state();
    }
}

RPY_EXTERN
void rpy_reverse_db_send_answer(int cmd, int64_t arg1, int64_t arg2,
                                int64_t arg3, RPyString *extra)
{
    rpy_revdb_command_t c;
    size_t extra_size = RPyString_Size(extra);
    c.cmd = cmd;
    c.extra_size = extra_size;
    if (c.extra_size != extra_size) {
        fprintf(stderr, "string too large (more than 4GB)\n");
        exit(1);
    }
    c.arg1 = arg1;
    c.arg2 = arg2;
    c.arg3 = arg3;
    write_sock(&c, sizeof(c));
    if (extra_size > 0)
        write_sock(_RPyString_AsString(extra), extra_size);
}

RPY_EXTERN
void rpy_reverse_db_breakpoint(int64_t num)
{
    if (flag_io_disabled != FID_REGULAR_MODE) {
        /* called from a debug command, ignore */
        return;
    }

    switch (breakpoint_mode) {
    case 'i':
        return;   /* ignored breakpoints */

    case 'r':     /* record the breakpoint but continue */
        if (last_recorded_breakpoint_loc != rpy_revdb.stop_point_seen) {
            last_recorded_breakpoint_loc = rpy_revdb.stop_point_seen;
            n_last_recorded_breakpoints = 0;
        }
        if (n_last_recorded_breakpoints < RECORD_BKPT_NUM) {
            last_recorded_breakpoint_nums[n_last_recorded_breakpoints] = num;
            n_last_recorded_breakpoints++;
        }
        return;

    case 'b':     /* default handling of breakpoints */
        interactive_break = rpy_revdb.stop_point_seen + 1;
        set_revdb_breakpoints();
        write_answer(ANSWER_BREAKPOINT, rpy_revdb.stop_point_break, 0, num);
        return;

    default:
        fprintf(stderr, "bad value %d of breakpoint_mode\n",
                (int)breakpoint_mode);
        exit(1);
    }
}

RPY_EXTERN
long long rpy_reverse_db_get_value(char value_id)
{
    switch (value_id) {
    case 'c':       /* current_time() */
        return (flag_io_disabled == FID_REGULAR_MODE ?
                rpy_revdb.stop_point_seen :
                saved_state.stop_point_seen);
    case 't':       /* total_time() */
        return total_stop_points;
    case 'b':       /* current_break_time() */
        return interactive_break;
    case 'u':       /* currently_created_objects() */
        return (flag_io_disabled == FID_REGULAR_MODE ?
                rpy_revdb.unique_id_seen :
                saved_state.unique_id_seen);
    default:
        return -1;
    }
}

RPY_EXTERN
uint64_t rpy_reverse_db_unique_id_break(void *new_object)
{
    uint64_t uid = rpy_revdb.unique_id_seen;
    bool_t watch_enabled = rpy_revdb.watch_enabled;

    if (!new_object) {
        fprintf(stderr, "out of memory: allocation failed, cannot continue\n");
        exit(1);
    }

    save_state();
    if (rpy_revdb_commands.rp_alloc) {
        protect_jmpbuf();
        if (setjmp(jmp_buf_cancel_execution) == 0)
            rpy_revdb_commands.rp_alloc(uid, new_object);
        unprotect_jmpbuf();
    }
    uid_break = *++future_next_id;
    restore_state();
    rpy_revdb.watch_enabled = watch_enabled;
    return uid;
}

struct destructor_s {
    void *d_obj;
    void (*d_callback)(void *);
};

static int _ftree_compare(const void *obj1, const void *obj2)
{
    const struct destructor_s *d1 = obj1;
    const struct destructor_s *d2 = obj2;
    struct pypy_header0 *h1 = d1->d_obj;
    struct pypy_header0 *h2 = d2->d_obj;
    if (h1->h_uid < h2->h_uid)
        return -1;
    if (h1->h_uid == h2->h_uid)
        return 0;
    else
        return 1;
}

static void _ftree_add(void **tree, void *obj, void (*callback)(void *))
{
    /* Note: we always allocate an indirection through a 
       struct destructor_s, so that Boehm knows that 'obj' must be
       kept alive. */
    struct destructor_s *node, **item;
    node = GC_MALLOC_UNCOLLECTABLE(sizeof(struct destructor_s));
    node->d_obj = obj;
    node->d_callback = callback;
    item = tsearch(node, tree, _ftree_compare);
    if (item == NULL) {
        fprintf(stderr, "_ftree_add: out of memory\n");
        exit(1);
    }
    if (*item != node) {
        fprintf(stderr, "_ftree_add: duplicate object\n");
        exit(1);
    }
}

static struct pypy_header0 *_ftree_pop(void **tree, uint64_t uid,
                                       void (**callback_out)(void *))
{
    struct destructor_s d_dummy, *entry, **item;
    struct pypy_header0 o_dummy, *result;

    d_dummy.d_obj = &o_dummy;
    o_dummy.h_uid = uid;
    item = tfind(&d_dummy, tree, _ftree_compare);
    if (item == NULL) {
        fprintf(stderr, "_ftree_pop: object not found\n");
        exit(1);
    }
    entry = *item;
    result = entry->d_obj;
    if (callback_out)
        *callback_out = entry->d_callback;
    assert(result->h_uid == uid);
    tdelete(entry, tree, _ftree_compare);
    GC_FREE(entry);
    return result;
}

RPY_EXTERN
int rpy_reverse_db_fq_register(void *obj)
{
    /*fprintf(stderr, "FINALIZER_TREE: %lld -> %p\n",
              ((struct pypy_header0 *)obj)->h_uid, obj);*/
    if (!RPY_RDB_REPLAY) {
        return 0;     /* recording */
    }
    else {
        /* add the object into the finalizer_tree, keyed by the h_uid. */
        _ftree_add(&finalizer_tree, obj, NULL);
        return 1;     /* replaying */
    }
}

RPY_EXTERN
void *rpy_reverse_db_next_dead(void *result)
{
    int64_t uid;
    RPY_REVDB_EMIT(uid = result ? ((struct pypy_header0 *)result)->h_uid : -1;,
                   int64_t _e, uid);
    /*fprintf(stderr, "next_dead: object %lld\n", uid);*/
    if (RPY_RDB_REPLAY) {
        if (uid == -1) {
            result = NULL;
        }
        else {
            /* fetch and remove the object from the finalizer_tree */
            result = _ftree_pop(&finalizer_tree, uid, NULL);
        }
    }
    return result;
}

RPY_EXTERN
void rpy_reverse_db_register_destructor(void *obj, void (*callback)(void *))
{
    if (!RPY_RDB_REPLAY) {
        GC_REGISTER_FINALIZER(obj, (GC_finalization_proc)callback,
                              NULL, NULL, NULL);
    }
    else {
        _ftree_add(&destructor_tree, obj, callback);
    }
}

static void replay_call_destructors(void)
{
    /* Re-enable fetching (disabled when we saw ASYNC_FINALIZER_TRIGGER),
       and fetch the uid's of dying objects with old-style destructors.
    */
    finalizer_break = -1;
    set_revdb_breakpoints();
    rpy_reverse_db_fetch(__FILE__, __LINE__);

    while (1) {
        int64_t uid;
        struct pypy_header0 *obj;
        void (*callback)(void *);

        RPY_REVDB_EMIT(abort();, int64_t _e, uid);
        if (uid == -1)
            break;

        obj = _ftree_pop(&destructor_tree, uid, &callback);
        callback(obj);
    }

    /* Now we're back in normal mode.  We trigger the finalizer 
       queues here. */
    fq_trigger();
}

static void *callbacklocs[] = {
    RPY_CALLBACKLOCS     /* macro from revdb_def.h */
};

RPY_EXTERN
void rpy_reverse_db_invoke_callback(unsigned char e)
{
    /* Replaying: we have read the byte which follows calls, expecting
       to see 0xFC, but we saw something else.  It's part of a two-bytes
       callback identifier. */

    do {
        unsigned long index;
        unsigned char e2;
        void (*pfn)(void);
        _RPY_REVDB_EMIT_REPLAY(unsigned char _e, e2)
        index = (e << 8) | e2;
        index -= 300;
        if (index >= (sizeof(callbacklocs) / sizeof(callbacklocs[0]))) {
            fprintf(stderr, "bad callback index\n");
            exit(1);
        }
        pfn = callbacklocs[index];
        pfn();

        _RPY_REVDB_EMIT_REPLAY(unsigned char _e, e)
    } while (e != 0xFC);
}


/* ------------------------------------------------------------ */


RPY_EXTERN
void seeing_uid(uint64_t uid)
{
}
