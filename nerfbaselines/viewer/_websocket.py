import struct
import hashlib
import os
import base64
import zlib
from enum import Enum, IntEnum
from collections import deque
from codecs import getincrementaldecoder
from typing import Optional, Union, Generator, Tuple, Deque, NamedTuple, Dict, Iterable, Generic, List, TypeVar
from dataclasses import dataclass, field
import selectors
from time import time

import h11


# The MIT License (MIT)
# 
# Copyright (c) 2017 Benno Rice and contributors
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.



class ConnectionType(Enum):
    """An enumeration of connection types."""

    #: This connection will act as client and talk to a remote server
    CLIENT = 1

    #: This connection will as as server and waits for client connections
    SERVER = 2


class ConnectionState(Enum):
    #: The opening handshake is in progress.
    CONNECTING = 0
    #: The opening handshake is complete.
    OPEN = 1
    #: The remote WebSocket has initiated a connection close.
    REMOTE_CLOSING = 2
    #: The local WebSocket (i.e. this instance) has initiated a connection close.
    LOCAL_CLOSING = 3
    #: The closing handshake has completed.
    CLOSED = 4
    #: The connection was rejected during the opening handshake.
    REJECTING = 5


class CloseReason(IntEnum):
    """
    RFC 6455, Section 7.4.1 - Defined Status Codes
    """

    #: indicates a normal closure, meaning that the purpose for
    #: which the connection was established has been fulfilled.
    NORMAL_CLOSURE = 1000

    #: indicates that an endpoint is "going away", such as a server
    #: going down or a browser having navigated away from a page.
    GOING_AWAY = 1001

    #: indicates that an endpoint is terminating the connection due
    #: to a protocol error.
    PROTOCOL_ERROR = 1002

    #: indicates that an endpoint is terminating the connection
    #: because it has received a type of data it cannot accept (e.g., an
    #: endpoint that understands only text data MAY send this if it
    #: receives a binary message).
    UNSUPPORTED_DATA = 1003

    #: Reserved.  The specific meaning might be defined in the future.
    # DON'T DEFINE THIS: RESERVED_1004 = 1004

    #: is a reserved value and MUST NOT be set as a status code in a
    #: Close control frame by an endpoint.  It is designated for use in
    #: applications expecting a status code to indicate that no status
    #: code was actually present.
    NO_STATUS_RCVD = 1005

    #: is a reserved value and MUST NOT be set as a status code in a
    #: Close control frame by an endpoint.  It is designated for use in
    #: applications expecting a status code to indicate that the
    #: connection was closed abnormally, e.g., without sending or
    #: receiving a Close control frame.
    ABNORMAL_CLOSURE = 1006

    #: indicates that an endpoint is terminating the connection
    #: because it has received data within a message that was not
    #: consistent with the type of the message (e.g., non-UTF-8 [RFC3629]
    #: data within a text message).
    INVALID_FRAME_PAYLOAD_DATA = 1007

    #: indicates that an endpoint is terminating the connection
    #: because it has received a message that violates its policy.  This
    #: is a generic status code that can be returned when there is no
    #: other more suitable status code (e.g., 1003 or 1009) or if there
    #: is a need to hide specific details about the policy.
    POLICY_VIOLATION = 1008

    #: indicates that an endpoint is terminating the connection
    #: because it has received a message that is too big for it to
    #: process.
    MESSAGE_TOO_BIG = 1009

    #: indicates that an endpoint (client) is terminating the
    #: connection because it has expected the server to negotiate one or
    #: more extension, but the server didn't return them in the response
    #: message of the WebSocket handshake.  The list of extensions that
    #: are needed SHOULD appear in the /reason/ part of the Close frame.
    #: Note that this status code is not used by the server, because it
    #: can fail the WebSocket handshake instead.
    MANDATORY_EXT = 1010

    #: indicates that a server is terminating the connection because
    #: it encountered an unexpected condition that prevented it from
    #: fulfilling the request.
    INTERNAL_ERROR = 1011

    #: Server/service is restarting
    #: (not part of RFC6455)
    SERVICE_RESTART = 1012

    #: Temporary server condition forced blocking client's request
    #: (not part of RFC6455)
    TRY_AGAIN_LATER = 1013

    #: is a reserved value and MUST NOT be set as a status code in a
    #: Close control frame by an endpoint.  It is designated for use in
    #: applications expecting a status code to indicate that the
    #: connection was closed due to a failure to perform a TLS handshake
    #: (e.g., the server certificate can't be verified).
    TLS_HANDSHAKE_FAILED = 1015


class Opcode(IntEnum):
    """
    RFC 6455, Section 5.2 - Base Framing Protocol
    """

    #: Continuation frame
    CONTINUATION = 0x0

    #: Text message
    TEXT = 0x1

    #: Binary message
    BINARY = 0x2

    #: Close frame
    CLOSE = 0x8

    #: Ping frame
    PING = 0x9

    #: Pong frame
    PONG = 0xA

    def iscontrol(self) -> bool:
        return bool(self & 0x08)


class Event:
    """
    Base class for wsproto events.
    """

    pass  # noqa


@dataclass(frozen=True)
class Request(Event):
    """The beginning of a Websocket connection, the HTTP Upgrade request

    This event is fired when a SERVER connection receives a WebSocket
    handshake request (HTTP with upgrade header).

    Fields:

    .. attribute:: host

       (Required) The hostname, or host header value.

    .. attribute:: target

       (Required) The request target (path and query string)

    .. attribute:: extensions

       The proposed extensions.

    .. attribute:: extra_headers

       The additional request headers, excluding extensions, host, subprotocols,
       and version headers.

    .. attribute:: subprotocols

       A list of the subprotocols proposed in the request, as a list
       of strings.
    """

    host: str
    target: str
    extensions: List = field(default_factory=list)
    extra_headers: List = field(default_factory=list)
    subprotocols: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AcceptConnection(Event):
    """The acceptance of a Websocket upgrade request.

    This event is fired when a CLIENT receives an acceptance response
    from a server. It is also used to accept an upgrade request when
    acting as a SERVER.

    Fields:

    .. attribute:: extra_headers

       Any additional (non websocket related) headers present in the
       acceptance response.

    .. attribute:: subprotocol

       The accepted subprotocol to use.

    """

    subprotocol: Optional[str] = None
    extensions: List = field(default_factory=list)
    extra_headers: List = field(default_factory=list)


@dataclass(frozen=True)
class RejectConnection(Event):
    """The rejection of a Websocket upgrade request, the HTTP response.

    The ``RejectConnection`` event sends the appropriate HTTP headers to
    communicate to the peer that the handshake has been rejected. You may also
    send an HTTP body by setting the ``has_body`` attribute to ``True`` and then
    sending one or more :class:`RejectData` events after this one. When sending
    a response body, the caller should set the ``Content-Length``,
    ``Content-Type``, and/or ``Transfer-Encoding`` headers as appropriate.

    When receiving a ``RejectConnection`` event, the ``has_body`` attribute will
    in almost all cases be ``True`` (even if the server set it to ``False``) and
    will be followed by at least one ``RejectData`` events, even though the data
    itself might be just ``b""``. (The only scenario in which the caller
    receives a ``RejectConnection`` with ``has_body == False`` is if the peer
    violates sends an informational status code (1xx) other than 101.)

    The ``has_body`` attribute should only be used when receiving the event. (It
    has ) is False the headers must include a
    content-length or transfer encoding.

    Fields:

    .. attribute:: headers (Headers)

       The headers to send with the response.

    .. attribute:: has_body

       This defaults to False, but set to True if there is a body. See
       also :class:`~RejectData`.

    .. attribute:: status_code

       The response status code.

    """

    status_code: int = 400
    headers: List = field(default_factory=list)
    has_body: bool = False


@dataclass(frozen=True)
class RejectData(Event):
    """The rejection HTTP response body.

    The caller may send multiple ``RejectData`` events. The final event should
    have the ``body_finished`` attribute set to ``True``.

    Fields:

    .. attribute:: body_finished

       True if this is the final chunk of the body data.

    .. attribute:: data (bytes)

       (Required) The raw body data.

    """

    data: bytes
    body_finished: bool = True


@dataclass(frozen=True)
class CloseConnection(Event):
    """The end of a Websocket connection, represents a closure frame.

    **wsproto does not automatically send a response to a close event.** To
    comply with the RFC you MUST send a close event back to the remote WebSocket
    if you have not already sent one. The :meth:`response` method provides a
    suitable event for this purpose, and you should check if a response needs
    to be sent by checking :func:`wsproto.WSConnection.state`.

    Fields:

    .. attribute:: code

       (Required) The integer close code to indicate why the connection
       has closed.

    .. attribute:: reason

       Additional reasoning for why the connection has closed.

    """

    code: int
    reason: Optional[str] = None

    def response(self) -> "CloseConnection":
        """Generate an RFC-compliant close frame to send back to the peer."""
        return CloseConnection(code=self.code, reason=self.reason)


T = TypeVar("T", bytes, str)


@dataclass(frozen=True)
class Message(Event, Generic[T]):
    """The websocket data message.

    Fields:

    .. attribute:: data

       (Required) The message data as byte string, can be decoded as UTF-8 for
       TEXT messages.  This only represents a single chunk of data and
       not a full WebSocket message.  You need to buffer and
       reassemble these chunks to get the full message.

    .. attribute:: frame_finished

       This has no semantic content, but is provided just in case some
       weird edge case user wants to be able to reconstruct the
       fragmentation pattern of the original stream.

    .. attribute:: message_finished

       True if this frame is the last one of this message, False if
       more frames are expected.

    """

    data: T
    frame_finished: bool = True
    message_finished: bool = True


@dataclass(frozen=True)
class TextMessage(Message[str]):  # pylint: disable=unsubscriptable-object
    """This event is fired when a data frame with TEXT payload is received.

    Fields:

    .. attribute:: data

       The message data as string, This only represents a single chunk
       of data and not a full WebSocket message.  You need to buffer
       and reassemble these chunks to get the full message.

    """

    # https://github.com/python/mypy/issues/5744
    data: str


@dataclass(frozen=True)
class BytesMessage(Message[bytes]):  # pylint: disable=unsubscriptable-object
    """This event is fired when a data frame with BINARY payload is
    received.

    Fields:

    .. attribute:: data

       The message data as byte string, can be decoded as UTF-8 for
       TEXT messages.  This only represents a single chunk of data and
       not a full WebSocket message.  You need to buffer and
       reassemble these chunks to get the full message.
    """

    # https://github.com/python/mypy/issues/5744
    data: bytes


@dataclass(frozen=True)
class Ping(Event):
    """The Ping event can be sent to trigger a ping frame and is fired
    when a Ping is received.

    **wsproto does not automatically send a pong response to a ping event.** To
    comply with the RFC you MUST send a pong even as soon as is practical. The
    :meth:`response` method provides a suitable event for this purpose.

    Fields:

    .. attribute:: payload

       An optional payload to emit with the ping frame.
    """

    payload: bytes = b""

    def response(self) -> "Pong":
        """Generate an RFC-compliant :class:`Pong` response to this ping."""
        return Pong(payload=self.payload)


@dataclass(frozen=True)
class Pong(Event):
    """The Pong event is fired when a Pong is received.

    Fields:

    .. attribute:: payload

       An optional payload to emit with the pong frame.

    """

    payload: bytes = b""


class ProtocolError(Exception):
    pass


class LocalProtocolError(ProtocolError):
    """Indicates an error due to local/programming errors.

    This is raised when the connection is asked to do something that
    is either incompatible with the state or the websocket standard.

    """

    pass  # noqa


class RemoteProtocolError(ProtocolError):
    """Indicates an error due to the remote's actions.

    This is raised when processing the bytes from the remote if the
    remote has sent data that is incompatible with the websocket
    standard.

    .. attribute:: event_hint

       This is a suggested wsproto Event to send to the client based
       on the error. It could be None if no hint is available.

    """

    def __init__(self, message: str, event_hint: Optional[Event] = None) -> None:
        self.event_hint = event_hint
        super().__init__(message)


class ParseFailed(Exception):
    def __init__(
        self, msg: str, code: CloseReason = CloseReason.PROTOCOL_ERROR
    ) -> None:
        super().__init__(msg)
        self.code = code


# RFC 6455, Section 7.4.1 - Defined Status Codes
LOCAL_ONLY_CLOSE_REASONS = (
    CloseReason.NO_STATUS_RCVD,
    CloseReason.ABNORMAL_CLOSURE,
    CloseReason.TLS_HANDSHAKE_FAILED,
)

# RFC 6455, Section 7.4.2 - Status Code Ranges
MIN_CLOSE_REASON = 1000
MIN_PROTOCOL_CLOSE_REASON = 1000
MAX_PROTOCOL_CLOSE_REASON = 2999
MIN_LIBRARY_CLOSE_REASON = 3000
MAX_LIBRARY_CLOSE_REASON = 3999
MIN_PRIVATE_CLOSE_REASON = 4000
MAX_PRIVATE_CLOSE_REASON = 4999
MAX_CLOSE_REASON = 4999

# RFC6455, Section 5.2 - Base Framing Protocol

# Payload length constants
PAYLOAD_LENGTH_TWO_BYTE = 126
PAYLOAD_LENGTH_EIGHT_BYTE = 127
MAX_PAYLOAD_NORMAL = 125
MAX_PAYLOAD_TWO_BYTE = 2**16 - 1
MAX_PAYLOAD_EIGHT_BYTE = 2**64 - 1
MAX_FRAME_PAYLOAD = MAX_PAYLOAD_EIGHT_BYTE

# MASK and PAYLOAD LEN are packed into a byte
MASK_MASK = 0x80
PAYLOAD_LEN_MASK = 0x7F

# FIN, RSV[123] and OPCODE are packed into a single byte
FIN_MASK = 0x80
RSV1_MASK = 0x40
RSV2_MASK = 0x20
RSV3_MASK = 0x10
OPCODE_MASK = 0x0F


NULL_MASK = struct.pack("!I", 0)


class RsvBits(NamedTuple):
    rsv1: bool
    rsv2: bool
    rsv3: bool


class Header(NamedTuple):
    fin: bool
    rsv: RsvBits
    opcode: Opcode
    payload_len: int
    masking_key: Optional[bytes]


class Frame(NamedTuple):
    opcode: Opcode
    payload: Union[bytes, str, Tuple[int, str]]
    frame_finished: bool
    message_finished: bool


def _truncate_utf8(data: bytes, nbytes: int) -> bytes:
    if len(data) <= nbytes:
        return data

    # Truncate
    data = data[:nbytes]
    # But we might have cut a codepoint in half, in which case we want to
    # discard the partial character so the data is at least
    # well-formed. This is a little inefficient since it processes the
    # whole message twice when in theory we could just peek at the last
    # few characters, but since this is only used for close messages (max
    # length = 125 bytes) it really doesn't matter.
    data = data.decode("utf-8", errors="ignore").encode("utf-8")
    return data


class Buffer:
    def __init__(self, initial_bytes: Optional[bytes] = None) -> None:
        self.buffer = bytearray()
        self.bytes_used = 0
        if initial_bytes:
            self.feed(initial_bytes)

    def feed(self, new_bytes: bytes) -> None:
        self.buffer += new_bytes

    def consume_at_most(self, nbytes: int) -> bytes:
        if not nbytes:
            return bytearray()

        data = self.buffer[self.bytes_used : self.bytes_used + nbytes]
        self.bytes_used += len(data)
        return data

    def consume_exactly(self, nbytes: int) -> Optional[bytes]:
        if len(self.buffer) - self.bytes_used < nbytes:
            return None

        return self.consume_at_most(nbytes)

    def commit(self) -> None:
        # In CPython 3.4+, del[:n] is amortized O(n), *not* quadratic
        del self.buffer[: self.bytes_used]
        self.bytes_used = 0

    def rollback(self) -> None:
        self.bytes_used = 0

    def __len__(self) -> int:
        return len(self.buffer)


class MessageDecoder:
    def __init__(self) -> None:
        self.opcode = None
        self.decoder = None

    def process_frame(self, frame: Frame) -> Frame:
        assert not frame.opcode.iscontrol()

        if self.opcode is None:
            if frame.opcode is Opcode.CONTINUATION:
                raise ParseFailed("unexpected CONTINUATION")
            self.opcode = frame.opcode
        elif frame.opcode is not Opcode.CONTINUATION:
            raise ParseFailed("expected CONTINUATION, got %r" % frame.opcode)

        if frame.opcode is Opcode.TEXT:
            self.decoder = getincrementaldecoder("utf-8")()

        finished = frame.frame_finished and frame.message_finished

        if self.decoder is None:
            data = frame.payload
        else:
            assert isinstance(frame.payload, (bytes, bytearray))
            try:
                data = self.decoder.decode(frame.payload, finished)
            except UnicodeDecodeError as exc:
                raise ParseFailed(str(exc), CloseReason.INVALID_FRAME_PAYLOAD_DATA)

        frame = Frame(self.opcode, data, frame.frame_finished, finished)

        if finished:
            self.opcode = None
            self.decoder = None

        return frame


CLIENT = ConnectionType.CLIENT
SERVER = ConnectionType.SERVER
_XOR_TABLE = [bytes(a ^ b for a in range(256)) for b in range(256)]


class XorMaskerSimple:
    def __init__(self, masking_key: bytes) -> None:
        self._masking_key = masking_key

    def process(self, data: bytes) -> bytes:
        if data:
            data_array = bytearray(data)
            a, b, c, d = (_XOR_TABLE[n] for n in self._masking_key)
            data_array[::4] = data_array[::4].translate(a)
            data_array[1::4] = data_array[1::4].translate(b)
            data_array[2::4] = data_array[2::4].translate(c)
            data_array[3::4] = data_array[3::4].translate(d)

            # Rotate the masking key so that the next usage continues
            # with the next key element, rather than restarting.
            key_rotation = len(data) % 4
            self._masking_key = (
                self._masking_key[key_rotation:] + self._masking_key[:key_rotation]
            )

            return bytes(data_array)
        return data


class XorMaskerNull:
    def process(self, data: bytes) -> bytes:
        return data


class FrameDecoder:
    def __init__(self, client: bool, extensions: Optional[List] = None) -> None:
        self.client = client
        self.extensions = extensions or []

        self.buffer = Buffer()

        self.header: Optional[Header] = None
        self.effective_opcode: Optional[Opcode] = None
        self.masker: Union[None, XorMaskerNull, XorMaskerSimple] = None
        self.payload_required = 0
        self.payload_consumed = 0

    def receive_bytes(self, data: bytes) -> None:
        self.buffer.feed(data)

    def process_buffer(self) -> Optional[Frame]:
        if not self.header:
            if not self.parse_header():
                return None
        # parse_header() sets these.
        assert self.header is not None
        assert self.masker is not None
        assert self.effective_opcode is not None

        if len(self.buffer) < self.payload_required:
            return None

        payload_remaining = self.header.payload_len - self.payload_consumed
        payload = self.buffer.consume_at_most(payload_remaining)
        if not payload and self.header.payload_len > 0:
            return None
        self.buffer.commit()

        self.payload_consumed += len(payload)
        finished = self.payload_consumed == self.header.payload_len

        payload = self.masker.process(payload)

        for extension in self.extensions:
            payload_ = extension.frame_inbound_payload_data(self, payload)
            if isinstance(payload_, CloseReason):
                raise ParseFailed("error in extension", payload_)
            payload = payload_

        if finished:
            final = bytearray()
            for extension in self.extensions:
                result = extension.frame_inbound_complete(self, self.header.fin)
                if isinstance(result, CloseReason):
                    raise ParseFailed("error in extension", result)
                if result is not None:
                    final += result
            payload += final

        frame = Frame(self.effective_opcode, payload, finished, self.header.fin)

        if finished:
            self.header = None
            self.effective_opcode = None
            self.masker = None
        else:
            self.effective_opcode = Opcode.CONTINUATION

        return frame

    def parse_header(self) -> bool:
        data = self.buffer.consume_exactly(2)
        if data is None:
            self.buffer.rollback()
            return False

        fin = bool(data[0] & FIN_MASK)
        rsv = RsvBits(
            bool(data[0] & RSV1_MASK),
            bool(data[0] & RSV2_MASK),
            bool(data[0] & RSV3_MASK),
        )
        opcode = data[0] & OPCODE_MASK
        try:
            opcode = Opcode(opcode)
        except ValueError:
            raise ParseFailed(f"Invalid opcode {opcode:#x}")

        if opcode.iscontrol() and not fin:
            raise ParseFailed("Invalid attempt to fragment control frame")

        has_mask = bool(data[1] & MASK_MASK)
        payload_len_short = data[1] & PAYLOAD_LEN_MASK
        payload_len = self.parse_extended_payload_length(opcode, payload_len_short)
        if payload_len is None:
            self.buffer.rollback()
            return False

        self.extension_processing(opcode, rsv, payload_len)

        if has_mask and self.client:
            raise ParseFailed("client received unexpected masked frame")
        if not has_mask and not self.client:
            raise ParseFailed("server received unexpected unmasked frame")
        if has_mask:
            masking_key = self.buffer.consume_exactly(4)
            if masking_key is None:
                self.buffer.rollback()
                return False
            self.masker = XorMaskerSimple(masking_key)
        else:
            self.masker = XorMaskerNull()

        self.buffer.commit()
        self.header = Header(fin, rsv, opcode, payload_len, None)
        self.effective_opcode = self.header.opcode
        if self.header.opcode.iscontrol():
            self.payload_required = payload_len
        else:
            self.payload_required = 0
        self.payload_consumed = 0
        return True

    def parse_extended_payload_length(
        self, opcode: Opcode, payload_len: int
    ) -> Optional[int]:
        if opcode.iscontrol() and payload_len > MAX_PAYLOAD_NORMAL:
            raise ParseFailed("Control frame with payload len > 125")
        if payload_len == PAYLOAD_LENGTH_TWO_BYTE:
            data = self.buffer.consume_exactly(2)
            if data is None:
                return None
            (payload_len,) = struct.unpack("!H", data)
            if payload_len <= MAX_PAYLOAD_NORMAL:
                raise ParseFailed(
                    "Payload length used 2 bytes when 1 would have sufficed"
                )
        elif payload_len == PAYLOAD_LENGTH_EIGHT_BYTE:
            data = self.buffer.consume_exactly(8)
            if data is None:
                return None
            (payload_len,) = struct.unpack("!Q", data)
            if payload_len <= MAX_PAYLOAD_TWO_BYTE:
                raise ParseFailed(
                    "Payload length used 8 bytes when 2 would have sufficed"
                )
            if payload_len >> 63:
                # I'm not sure why this is illegal, but that's what the RFC
                # says, so...
                raise ParseFailed("8-byte payload length with non-zero MSB")

        return payload_len

    def extension_processing(
        self, opcode: Opcode, rsv: RsvBits, payload_len: int
    ) -> None:
        rsv_used = [False, False, False]
        for extension in self.extensions:
            result = extension.frame_inbound_header(self, opcode, rsv, payload_len)
            if isinstance(result, CloseReason):
                raise ParseFailed("error in extension", result)
            for bit, used in enumerate(result):
                if used:
                    rsv_used[bit] = True
        for expected, found in zip(rsv_used, rsv):
            if found and not expected:
                raise ParseFailed("Reserved bit set unexpectedly")



class FrameProtocol:
    def __init__(self, client: bool, extensions: List) -> None:
        self.client = client
        self.extensions = [ext for ext in extensions if ext.enabled()]

        # Global state
        self._frame_decoder = FrameDecoder(self.client, self.extensions)
        self._message_decoder = MessageDecoder()
        self._parse_more = self._parse_more_gen()

        self._outbound_opcode: Optional[Opcode] = None

    def _process_close(self, frame: Frame) -> Frame:
        data = frame.payload
        assert isinstance(data, (bytes, bytearray))

        if not data:
            # "If this Close control frame contains no status code, _The
            # WebSocket Connection Close Code_ is considered to be 1005"
            data = (CloseReason.NO_STATUS_RCVD, "")
        elif len(data) == 1:
            raise ParseFailed("CLOSE with 1 byte payload")
        else:
            (code,) = struct.unpack("!H", data[:2])
            if code < MIN_CLOSE_REASON or code > MAX_CLOSE_REASON:
                raise ParseFailed("CLOSE with invalid code")
            try:
                code = CloseReason(code)
            except ValueError:
                pass
            if code in LOCAL_ONLY_CLOSE_REASONS:
                raise ParseFailed("remote CLOSE with local-only reason")
            if not isinstance(code, CloseReason) and code <= MAX_PROTOCOL_CLOSE_REASON:
                raise ParseFailed("CLOSE with unknown reserved code")
            try:
                reason = data[2:].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ParseFailed(
                    "Error decoding CLOSE reason: " + str(exc),
                    CloseReason.INVALID_FRAME_PAYLOAD_DATA,
                )
            data = (code, reason)

        return Frame(frame.opcode, data, frame.frame_finished, frame.message_finished)

    def _parse_more_gen(self) -> Generator[Optional[Frame], None, None]:
        # Consume as much as we can from self._buffer, yielding events, and
        # then yield None when we need more data. Or raise ParseFailed.

        # XX FIXME this should probably be refactored so that we never see
        # disabled extensions in the first place...
        self.extensions = [ext for ext in self.extensions if ext.enabled()]
        closed = False

        while not closed:
            frame = self._frame_decoder.process_buffer()

            if frame is not None:
                if not frame.opcode.iscontrol():
                    frame = self._message_decoder.process_frame(frame)
                elif frame.opcode == Opcode.CLOSE:
                    frame = self._process_close(frame)
                    closed = True

            yield frame

    def receive_bytes(self, data: bytes) -> None:
        self._frame_decoder.receive_bytes(data)

    def received_frames(self) -> Generator[Frame, None, None]:
        for event in self._parse_more:
            if event is None:
                break
            else:
                yield event

    def close(self, code: Optional[int] = None, reason: Optional[str] = None) -> bytes:
        payload = bytearray()
        if code is CloseReason.NO_STATUS_RCVD:
            code = None
        if code is None and reason:
            raise TypeError("cannot specify a reason without a code")
        if code in LOCAL_ONLY_CLOSE_REASONS:
            code = CloseReason.NORMAL_CLOSURE
        if code is not None:
            payload += bytearray(struct.pack("!H", code))
            if reason is not None:
                payload += _truncate_utf8(
                    reason.encode("utf-8"), MAX_PAYLOAD_NORMAL - 2
                )

        return self._serialize_frame(Opcode.CLOSE, payload)

    def ping(self, payload: bytes = b"") -> bytes:
        return self._serialize_frame(Opcode.PING, payload)

    def pong(self, payload: bytes = b"") -> bytes:
        return self._serialize_frame(Opcode.PONG, payload)

    def send_data(
        self, payload: Union[bytes, bytearray, str] = b"", fin: bool = True
    ) -> bytes:
        if isinstance(payload, (bytes, bytearray, memoryview)):
            opcode = Opcode.BINARY
        elif isinstance(payload, str):
            opcode = Opcode.TEXT
            payload = payload.encode("utf-8")
        else:
            raise ValueError("Must provide bytes or text")

        if self._outbound_opcode is None:
            self._outbound_opcode = opcode
        elif self._outbound_opcode is not opcode:
            raise TypeError("Data type mismatch inside message")
        else:
            opcode = Opcode.CONTINUATION

        if fin:
            self._outbound_opcode = None

        return self._serialize_frame(opcode, payload, fin)

    def _make_fin_rsv_opcode(self, fin: bool, rsv: RsvBits, opcode: Opcode) -> int:
        fin_bits = int(fin) << 7
        rsv_bits = (int(rsv.rsv1) << 6) + (int(rsv.rsv2) << 5) + (int(rsv.rsv3) << 4)
        opcode_bits = int(opcode)

        return fin_bits | rsv_bits | opcode_bits

    def _serialize_frame(
        self, opcode: Opcode, payload: bytes = b"", fin: bool = True
    ) -> bytes:
        rsv = RsvBits(False, False, False)
        for extension in reversed(self.extensions):
            rsv, payload = extension.frame_outbound(self, opcode, rsv, payload, fin)

        fin_rsv_opcode = self._make_fin_rsv_opcode(fin, rsv, opcode)

        payload_length = len(payload)
        quad_payload = False
        if payload_length <= MAX_PAYLOAD_NORMAL:
            first_payload = payload_length
            second_payload = None
        elif payload_length <= MAX_PAYLOAD_TWO_BYTE:
            first_payload = PAYLOAD_LENGTH_TWO_BYTE
            second_payload = payload_length
        else:
            first_payload = PAYLOAD_LENGTH_EIGHT_BYTE
            second_payload = payload_length
            quad_payload = True

        if self.client:
            first_payload |= 1 << 7

        header = bytearray([fin_rsv_opcode, first_payload])
        if second_payload is not None:
            if opcode.iscontrol():
                raise ValueError("payload too long for control frame")
            if quad_payload:
                header += bytearray(struct.pack("!Q", second_payload))
            else:
                header += bytearray(struct.pack("!H", second_payload))

        if self.client:
            # "The masking key is a 32-bit value chosen at random by the
            # client.  When preparing a masked frame, the client MUST pick a
            # fresh masking key from the set of allowed 32-bit values.  The
            # masking key needs to be unpredictable; thus, the masking key
            # MUST be derived from a strong source of entropy, and the masking
            # key for a given frame MUST NOT make it simple for a server/proxy
            # to predict the masking key for a subsequent frame.  The
            # unpredictability of the masking key is essential to prevent
            # authors of malicious applications from selecting the bytes that
            # appear on the wire."
            #   -- https://tools.ietf.org/html/rfc6455#section-5.3
            masking_key = os.urandom(4)
            masker = XorMaskerSimple(masking_key)
            return header + masking_key + masker.process(payload)

        return header + payload


class Connection:
    """
    A low-level WebSocket connection object.

    This wraps two other protocol objects, an HTTP/1.1 protocol object used
    to do the initial HTTP upgrade handshake and a WebSocket frame protocol
    object used to exchange messages and other control frames.
    """

    def __init__(
        self,
        connection_type: ConnectionType,
        extensions = None,
        trailing_data: bytes = b"",
    ) -> None:
        """
        Constructor

        :param wsproto.connection.ConnectionType connection_type: Whether this
            object is on the client- or server-side of a connection.
            To initialise as a client pass ``CLIENT`` otherwise pass ``SERVER``.
        :param list extensions: The proposed extensions.
        :param bytes trailing_data: Data that has been received, but not yet
            processed.
        """
        self.client = connection_type is ConnectionType.CLIENT
        self._events: Deque[Event] = deque()
        self._proto = FrameProtocol(self.client, extensions or [])
        self._state = ConnectionState.OPEN
        self.receive_data(trailing_data)

    @property
    def state(self) -> ConnectionState:
        return self._state

    def send(self, event: Event) -> bytes:
        data = b""
        if isinstance(event, Message) and self.state == ConnectionState.OPEN:
            data += self._proto.send_data(event.data, event.message_finished)
        elif isinstance(event, Ping) and self.state == ConnectionState.OPEN:
            data += self._proto.ping(event.payload)
        elif isinstance(event, Pong) and self.state == ConnectionState.OPEN:
            data += self._proto.pong(event.payload)
        elif isinstance(event, CloseConnection) and self.state in {
            ConnectionState.OPEN,
            ConnectionState.REMOTE_CLOSING,
        }:
            data += self._proto.close(event.code, event.reason)
            if self.state == ConnectionState.REMOTE_CLOSING:
                self._state = ConnectionState.CLOSED
            else:
                self._state = ConnectionState.LOCAL_CLOSING
        else:
            raise LocalProtocolError(
                f"Event {event} cannot be sent in state {self.state}."
            )
        return data

    def receive_data(self, data: Optional[bytes]) -> None:
        """
        Pass some received data to the connection for handling.

        A list of events that the remote peer triggered by sending this data can
        be retrieved with :meth:`~wsproto.connection.Connection.events`.

        :param data: The data received from the remote peer on the network.
        :type data: ``bytes``
        """

        if data is None:
            # "If _The WebSocket Connection is Closed_ and no Close control
            # frame was received by the endpoint (such as could occur if the
            # underlying transport connection is lost), _The WebSocket
            # Connection Close Code_ is considered to be 1006."
            self._events.append(CloseConnection(code=CloseReason.ABNORMAL_CLOSURE))
            self._state = ConnectionState.CLOSED
            return

        if self.state in (ConnectionState.OPEN, ConnectionState.LOCAL_CLOSING):
            self._proto.receive_bytes(data)
        elif self.state is ConnectionState.CLOSED:
            raise LocalProtocolError("Connection already closed.")
        else:
            pass  # pragma: no cover

    def events(self) -> Generator[Event, None, None]:
        """
        Return a generator that provides any events that have been generated
        by protocol activity.

        :returns: generator of :class:`Event <wsproto.events.Event>` subclasses
        """
        while self._events:
            yield self._events.popleft()

        try:
            for frame in self._proto.received_frames():
                if frame.opcode is Opcode.PING:
                    assert frame.frame_finished and frame.message_finished
                    assert isinstance(frame.payload, (bytes, bytearray))
                    yield Ping(payload=frame.payload)

                elif frame.opcode is Opcode.PONG:
                    assert frame.frame_finished and frame.message_finished
                    assert isinstance(frame.payload, (bytes, bytearray))
                    yield Pong(payload=frame.payload)

                elif frame.opcode is Opcode.CLOSE:
                    assert isinstance(frame.payload, tuple)
                    code, reason = frame.payload
                    if self.state is ConnectionState.LOCAL_CLOSING:
                        self._state = ConnectionState.CLOSED
                    else:
                        self._state = ConnectionState.REMOTE_CLOSING
                    yield CloseConnection(code=code, reason=reason)

                elif frame.opcode is Opcode.TEXT:
                    assert isinstance(frame.payload, str)
                    yield TextMessage(
                        data=frame.payload,
                        frame_finished=frame.frame_finished,
                        message_finished=frame.message_finished,
                    )

                elif frame.opcode is Opcode.BINARY:
                    assert isinstance(frame.payload, (bytes, bytearray))
                    yield BytesMessage(
                        data=frame.payload,
                        frame_finished=frame.frame_finished,
                        message_finished=frame.message_finished,
                    )

                else:
                    pass  # pragma: no cover
        except ParseFailed as exc:
            yield CloseConnection(code=exc.code, reason=str(exc))


# RFC6455, Section 1.3 - Opening Handshake
ACCEPT_GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# RFC6455, Section 4.2.1/6 - Reading the Client's Opening Handshake
WEBSOCKET_VERSION = b"13"

# RFC6455, Section 4.2.1/3 - Value of the Upgrade header
WEBSOCKET_UPGRADE = b"websocket"


class PerMessageDeflate:
    name = "permessage-deflate"

    DEFAULT_CLIENT_MAX_WINDOW_BITS = 15
    DEFAULT_SERVER_MAX_WINDOW_BITS = 15

    def __init__(
        self,
        client_no_context_takeover: bool = False,
        client_max_window_bits: Optional[int] = None,
        server_no_context_takeover: bool = False,
        server_max_window_bits: Optional[int] = None,
    ) -> None:
        self.client_no_context_takeover = client_no_context_takeover
        self.server_no_context_takeover = server_no_context_takeover
        self._client_max_window_bits = self.DEFAULT_CLIENT_MAX_WINDOW_BITS
        self._server_max_window_bits = self.DEFAULT_SERVER_MAX_WINDOW_BITS
        if client_max_window_bits is not None:
            self.client_max_window_bits = client_max_window_bits
        if server_max_window_bits is not None:
            self.server_max_window_bits = server_max_window_bits

        self._compressor = None  # noqa
        self._decompressor = None  # noqa
        # This refers to the current frame
        self._inbound_is_compressible: Optional[bool] = None
        # This refers to the ongoing message (which might span multiple
        # frames). Only the first frame in a fragmented message is flagged for
        # compression, so this carries that bit forward.
        self._inbound_compressed: Optional[bool] = None

        self._enabled = False

    @property
    def client_max_window_bits(self) -> int:
        return self._client_max_window_bits

    @client_max_window_bits.setter
    def client_max_window_bits(self, value: int) -> None:
        if value < 9 or value > 15:
            raise ValueError("Window size must be between 9 and 15 inclusive")
        self._client_max_window_bits = value

    @property
    def server_max_window_bits(self) -> int:
        return self._server_max_window_bits

    @server_max_window_bits.setter
    def server_max_window_bits(self, value: int) -> None:
        if value < 9 or value > 15:
            raise ValueError("Window size must be between 9 and 15 inclusive")
        self._server_max_window_bits = value

    def _compressible_opcode(self, opcode) -> bool:
        return opcode in (Opcode.TEXT, Opcode.BINARY, Opcode.CONTINUATION)

    def enabled(self) -> bool:
        return self._enabled

    def offer(self) -> Union[bool, str]:
        parameters = [
            "client_max_window_bits=%d" % self.client_max_window_bits,
            "server_max_window_bits=%d" % self.server_max_window_bits,
        ]

        if self.client_no_context_takeover:
            parameters.append("client_no_context_takeover")
        if self.server_no_context_takeover:
            parameters.append("server_no_context_takeover")

        return "; ".join(parameters)

    def finalize(self, offer: str) -> None:
        bits = [b.strip() for b in offer.split(";")]
        for bit in bits[1:]:
            if bit.startswith("client_no_context_takeover"):
                self.client_no_context_takeover = True
            elif bit.startswith("server_no_context_takeover"):
                self.server_no_context_takeover = True
            elif bit.startswith("client_max_window_bits"):
                self.client_max_window_bits = int(bit.split("=", 1)[1].strip())
            elif bit.startswith("server_max_window_bits"):
                self.server_max_window_bits = int(bit.split("=", 1)[1].strip())

        self._enabled = True

    def _parse_params(self, params: str):
        client_max_window_bits = None
        server_max_window_bits = None

        bits = [b.strip() for b in params.split(";")]
        for bit in bits[1:]:
            if bit.startswith("client_no_context_takeover"):
                self.client_no_context_takeover = True
            elif bit.startswith("server_no_context_takeover"):
                self.server_no_context_takeover = True
            elif bit.startswith("client_max_window_bits"):
                if "=" in bit:
                    client_max_window_bits = int(bit.split("=", 1)[1].strip())
                else:
                    client_max_window_bits = self.client_max_window_bits
            elif bit.startswith("server_max_window_bits"):
                if "=" in bit:
                    server_max_window_bits = int(bit.split("=", 1)[1].strip())
                else:
                    server_max_window_bits = self.server_max_window_bits

        return client_max_window_bits, server_max_window_bits

    def accept(self, offer: str) -> Union[bool, None, str]:
        client_max_window_bits, server_max_window_bits = self._parse_params(offer)

        parameters = []

        if self.client_no_context_takeover:
            parameters.append("client_no_context_takeover")
        if self.server_no_context_takeover:
            parameters.append("server_no_context_takeover")
        try:
            if client_max_window_bits is not None:
                parameters.append("client_max_window_bits=%d" % client_max_window_bits)
                self.client_max_window_bits = client_max_window_bits
            if server_max_window_bits is not None:
                parameters.append("server_max_window_bits=%d" % server_max_window_bits)
                self.server_max_window_bits = server_max_window_bits
        except ValueError:
            return None
        else:
            self._enabled = True
            return "; ".join(parameters)

    def frame_inbound_header(self, proto, opcode, rsv, payload_length):
        del payload_length
        if rsv.rsv1 and opcode.iscontrol():
            return CloseReason.PROTOCOL_ERROR
        if rsv.rsv1 and opcode is Opcode.CONTINUATION:
            return CloseReason.PROTOCOL_ERROR

        self._inbound_is_compressible = self._compressible_opcode(opcode)

        if self._inbound_compressed is None:
            self._inbound_compressed = rsv.rsv1
            if self._inbound_compressed:
                assert self._inbound_is_compressible
                if proto.client:
                    bits = self.server_max_window_bits
                else:
                    bits = self.client_max_window_bits
                if self._decompressor is None:
                    self._decompressor = zlib.decompressobj(-int(bits))

        return RsvBits(True, False, False)

    def frame_inbound_payload_data(self, proto, data: bytes) -> Union[bytes, CloseReason]:
        del proto
        if not self._inbound_compressed or not self._inbound_is_compressible:
            return data
        assert self._decompressor is not None

        try:
            return self._decompressor.decompress(bytes(data))
        except zlib.error:
            return CloseReason.INVALID_FRAME_PAYLOAD_DATA

    def frame_inbound_complete(self, proto, fin: bool) -> Union[bytes, CloseReason, None]:
        if not fin:
            return None
        if not self._inbound_is_compressible:
            self._inbound_compressed = None
            return None
        if not self._inbound_compressed:
            self._inbound_compressed = None
            return None
        assert self._decompressor is not None

        try:
            data = self._decompressor.decompress(b"\x00\x00\xff\xff")
            data += self._decompressor.flush()
        except zlib.error:
            return CloseReason.INVALID_FRAME_PAYLOAD_DATA

        if proto.client:
            no_context_takeover = self.server_no_context_takeover
        else:
            no_context_takeover = self.client_no_context_takeover

        if no_context_takeover:
            self._decompressor = None

        self._inbound_compressed = None

        return data

    def frame_outbound(self, proto, opcode: Opcode, rsv: RsvBits, data: bytes, fin: bool) -> Tuple[RsvBits, bytes]:
        if not self._compressible_opcode(opcode):
            return (rsv, data)

        if opcode is not Opcode.CONTINUATION:
            rsv = RsvBits(True, *rsv[1:])

        if self._compressor is None:
            assert opcode is not Opcode.CONTINUATION
            if proto.client:
                bits = self.client_max_window_bits
            else:
                bits = self.server_max_window_bits
            self._compressor = zlib.compressobj(
                zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -int(bits)
            )

        data = self._compressor.compress(bytes(data))

        if fin:
            data += self._compressor.flush(zlib.Z_SYNC_FLUSH)
            data = data[:-4]

            if proto.client:
                no_context_takeover = self.client_no_context_takeover
            else:
                no_context_takeover = self.server_no_context_takeover

            if no_context_takeover:
                self._compressor = None

        return (rsv, data)

    def __repr__(self) -> str:
        descr = ["client_max_window_bits=%d" % self.client_max_window_bits]
        if self.client_no_context_takeover:
            descr.append("client_no_context_takeover")
        descr.append("server_max_window_bits=%d" % self.server_max_window_bits)
        if self.server_no_context_takeover:
            descr.append("server_no_context_takeover")

        return "<{} {}>".format(self.__class__.__name__, "; ".join(descr))


def _split_comma_header(value: bytes) -> List[str]:
    return [piece.decode("ascii").strip() for piece in value.split(b",")]


def server_extensions_handshake(requested: Iterable[str], supported) -> Optional[bytes]:
    """Agree on the extensions to use returning an appropriate header value.

    This returns None if there are no agreed extensions
    """
    accepts: Dict[str, Union[bool, bytes]] = {}
    for offer in requested:
        name = offer.split(";", 1)[0].strip()
        for extension in supported:
            if extension.name == name:
                accept = extension.accept(offer)
                if isinstance(accept, bool):
                    if accept:
                        accepts[extension.name] = True
                elif accept is not None:
                    accepts[extension.name] = accept.encode("ascii")

    if accepts:
        extensions: List[bytes] = []
        for name, params in accepts.items():
            name_bytes = name.encode("ascii")
            if isinstance(params, bool):
                assert params
                extensions.append(name_bytes)
            else:
                if params == b"":
                    extensions.append(b"%s" % (name_bytes))
                else:
                    extensions.append(b"%s; %s" % (name_bytes, params))
        return b", ".join(extensions)

    return None


def client_extensions_handshake(accepted: Iterable[str], supported):
    # This raises RemoteProtocolError is the accepted extension is not
    # supported.
    extensions = []
    for accept in accepted:
        name = accept.split(";", 1)[0].strip()
        for extension in supported:
            if extension.name == name:
                extension.finalize(accept)
                extensions.append(extension)
                break
        else:
            raise RemoteProtocolError(
                f"unrecognized extension {name}", event_hint=RejectConnection()
            )
    return extensions


class H11Handshake:
    """A Handshake implementation for HTTP/1.1 connections."""

    def __init__(self, connection_type: ConnectionType) -> None:
        self.client = connection_type is ConnectionType.CLIENT
        self._state = ConnectionState.CONNECTING

        if self.client:
            self._h11_connection = h11.Connection(h11.CLIENT)
        else:
            self._h11_connection = h11.Connection(h11.SERVER)

        self._connection: Optional[Connection] = None
        self._events: Deque[Event] = deque()
        self._initiating_request: Optional[Request] = None
        self._nonce: Optional[bytes] = None

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def connection(self) -> Optional[Connection]:
        """Return the established connection.

        This will either return the connection or raise a
        LocalProtocolError if the connection has not yet been
        established.

        :rtype: h11.Connection
        """
        return self._connection

    def send(self, event: Event) -> bytes:
        """Send an event to the remote.

        This will return the bytes to send based on the event or raise
        a LocalProtocolError if the event is not valid given the
        state.

        :returns: Data to send to the WebSocket peer.
        :rtype: bytes
        """
        data = b""
        if isinstance(event, Request):
            data += self._initiate_connection(event)
        elif isinstance(event, AcceptConnection):
            data += self._accept(event)
        elif isinstance(event, RejectConnection):
            data += self._reject(event)
        elif isinstance(event, RejectData):
            data += self._send_reject_data(event)
        else:
            raise LocalProtocolError(
                f"Event {event} cannot be sent during the handshake"
            )
        return data

    def receive_data(self, data: Optional[bytes]) -> None:
        """Receive data from the remote.

        A list of events that the remote peer triggered by sending
        this data can be retrieved with :meth:`events`.

        :param bytes data: Data received from the WebSocket peer.
        """
        self._h11_connection.receive_data(data or b"")
        while True:
            try:
                event = self._h11_connection.next_event()
            except h11.RemoteProtocolError:
                raise RemoteProtocolError(
                    "Bad HTTP message", event_hint=RejectConnection()
                )
            if (
                isinstance(event, h11.ConnectionClosed)
                or event is h11.NEED_DATA
                or event is h11.PAUSED
            ):
                break

            if self.client:
                if isinstance(event, h11.InformationalResponse):
                    if event.status_code == 101:
                        self._events.append(self._establish_client_connection(event))
                    else:
                        self._events.append(
                            RejectConnection(
                                headers=list(event.headers),
                                status_code=event.status_code,
                                has_body=False,
                            )
                        )
                        self._state = ConnectionState.CLOSED
                elif isinstance(event, h11.Response):
                    self._state = ConnectionState.REJECTING
                    self._events.append(
                        RejectConnection(
                            headers=list(event.headers),
                            status_code=event.status_code,
                            has_body=True,
                        )
                    )
                elif isinstance(event, h11.Data):
                    self._events.append(
                        RejectData(data=event.data, body_finished=False)
                    )
                elif isinstance(event, h11.EndOfMessage):
                    self._events.append(RejectData(data=b"", body_finished=True))
                    self._state = ConnectionState.CLOSED
            else:
                if isinstance(event, h11.Request):
                    self._events.append(self._process_connection_request(event))

    def events(self) -> Generator[Event, None, None]:
        """Return a generator that provides any events that have been generated
        by protocol activity.

        :returns: a generator that yields H11 events.
        """
        while self._events:
            yield self._events.popleft()

    # Server mode methods

    def _process_connection_request(  # noqa: MC0001
        self, event: h11.Request
    ) -> Request:
        if event.method != b"GET":
            raise RemoteProtocolError(
                "Request method must be GET", event_hint=RejectConnection()
            )

        connection_tokens = None
        extensions: List[str] = []
        host = None
        key = None
        subprotocols: List[str] = []
        upgrade = b""
        version = None
        headers = []
        for name, value in event.headers:
            name = name.lower()
            if name == b"connection":
                connection_tokens = _split_comma_header(value)
            elif name == b"host":
                host = value.decode("idna")
                continue  # Skip appending to headers
            elif name == b"sec-websocket-extensions":
                extensions.extend(_split_comma_header(value))
                continue  # Skip appending to headers
            elif name == b"sec-websocket-key":
                key = value
            elif name == b"sec-websocket-protocol":
                subprotocols.extend(_split_comma_header(value))
                continue  # Skip appending to headers
            elif name == b"sec-websocket-version":
                version = value
            elif name == b"upgrade":
                upgrade = value
            headers.append((name, value))
        if connection_tokens is None or not any(
            token.lower() == "upgrade" for token in connection_tokens
        ):
            raise RemoteProtocolError(
                "Missing header, 'Connection: Upgrade'", event_hint=RejectConnection()
            )
        if version != WEBSOCKET_VERSION:
            raise RemoteProtocolError(
                "Missing header, 'Sec-WebSocket-Version'",
                event_hint=RejectConnection(
                    headers=[(b"Sec-WebSocket-Version", WEBSOCKET_VERSION)],
                    status_code=426 if version else 400,
                ),
            )
        if key is None:
            raise RemoteProtocolError(
                "Missing header, 'Sec-WebSocket-Key'", event_hint=RejectConnection()
            )
        if upgrade.lower() != WEBSOCKET_UPGRADE:
            raise RemoteProtocolError(
                f"Missing header, 'Upgrade: {WEBSOCKET_UPGRADE.decode()}'",
                event_hint=RejectConnection(),
            )
        if host is None:
            raise RemoteProtocolError(
                "Missing header, 'Host'", event_hint=RejectConnection()
            )

        self._initiating_request = Request(
            extensions=extensions,
            extra_headers=headers,
            host=host,
            subprotocols=subprotocols,
            target=event.target.decode("ascii"),
        )
        return self._initiating_request

    def _accept(self, event: AcceptConnection) -> bytes:
        # _accept is always called after _process_connection_request.
        assert self._initiating_request is not None

        request_headers_ = {}
        for name, value in self._initiating_request.extra_headers:
            request_headers_.setdefault(name, []).append(value)
        request_headers = {}
        for name, values in request_headers_.items():
            request_headers[name] = b", ".join(values)

        nonce = request_headers[b"sec-websocket-key"]
        accept_token = base64.b64encode(
            hashlib.sha1(nonce + ACCEPT_GUID).digest())

        headers = [
            (b"Upgrade", WEBSOCKET_UPGRADE),
            (b"Connection", b"Upgrade"),
            (b"Sec-WebSocket-Accept", accept_token),
        ]

        if event.subprotocol is not None:
            if event.subprotocol not in self._initiating_request.subprotocols:
                raise LocalProtocolError(f"unexpected subprotocol {event.subprotocol}")
            headers.append(
                (b"Sec-WebSocket-Protocol", event.subprotocol.encode("ascii"))
            )

        if event.extensions:
            accepts = server_extensions_handshake(
                self._initiating_request.extensions,
                event.extensions,
            )
            if accepts:
                headers.append((b"Sec-WebSocket-Extensions", accepts))

        response = h11.InformationalResponse(
            status_code=101, headers=headers + event.extra_headers
        )
        self._connection = Connection(
            ConnectionType.CLIENT if self.client else ConnectionType.SERVER,
            event.extensions,
        )
        self._state = ConnectionState.OPEN
        return self._h11_connection.send(response) or b""

    def _reject(self, event: RejectConnection) -> bytes:
        if self.state != ConnectionState.CONNECTING:
            raise LocalProtocolError(
                "Connection cannot be rejected in state %s" % self.state
            )

        headers = list(event.headers)
        if not event.has_body:
            headers.append((b"content-length", b"0"))
        response = h11.Response(status_code=event.status_code, headers=headers)
        data = self._h11_connection.send(response) or b""
        self._state = ConnectionState.REJECTING
        if not event.has_body:
            data += self._h11_connection.send(h11.EndOfMessage()) or b""
            self._state = ConnectionState.CLOSED
        return data

    def _send_reject_data(self, event: RejectData) -> bytes:
        if self.state != ConnectionState.REJECTING:
            raise LocalProtocolError(
                f"Cannot send rejection data in state {self.state}"
            )

        data = self._h11_connection.send(h11.Data(data=event.data)) or b""
        if event.body_finished:
            data += self._h11_connection.send(h11.EndOfMessage()) or b""
            self._state = ConnectionState.CLOSED
        return data

    # Client mode methods

    def _initiate_connection(self, request: Request) -> bytes:
        self._initiating_request = request
        self._nonce = base64.b64encode(os.urandom(16))

        headers = [
            (b"Host", request.host.encode("idna")),
            (b"Upgrade", WEBSOCKET_UPGRADE),
            (b"Connection", b"Upgrade"),
            (b"Sec-WebSocket-Key", self._nonce),
            (b"Sec-WebSocket-Version", WEBSOCKET_VERSION),
        ]

        if request.subprotocols:
            headers.append(
                (
                    b"Sec-WebSocket-Protocol",
                    (", ".join(request.subprotocols)).encode("ascii"),
                )
            )

        if request.extensions:
            offers: Dict[str, Union[str, bool]] = {}
            for e in request.extensions:
                offers[e.name] = e.offer()
            extensions = []
            for name, params in offers.items():
                bname = name.encode("ascii")
                if isinstance(params, bool):
                    if params:
                        extensions.append(bname)
                else:
                    extensions.append(b"%s; %s" % (bname, params.encode("ascii")))
            if extensions:
                headers.append((b"Sec-WebSocket-Extensions", b", ".join(extensions)))

        upgrade = h11.Request(
            method=b"GET",
            target=request.target.encode("ascii"),
            headers=headers + request.extra_headers,
        )
        return self._h11_connection.send(upgrade) or b""

    def _establish_client_connection(
        self, event: h11.InformationalResponse
    ) -> AcceptConnection:  # noqa: MC0001
        # _establish_client_connection is always called after _initiate_connection.
        assert self._initiating_request is not None
        assert self._nonce is not None

        accept = None
        connection_tokens = None
        accepts: List[str] = []
        subprotocol = None
        upgrade = b""
        headers = []
        for name, value in event.headers:
            name = name.lower()
            if name == b"connection":
                connection_tokens = _split_comma_header(value)
                continue  # Skip appending to headers
            elif name == b"sec-websocket-extensions":
                accepts = _split_comma_header(value)
                continue  # Skip appending to headers
            elif name == b"sec-websocket-accept":
                accept = value
                continue  # Skip appending to headers
            elif name == b"sec-websocket-protocol":
                subprotocol = value.decode("ascii")
                continue  # Skip appending to headers
            elif name == b"upgrade":
                upgrade = value
                continue  # Skip appending to headers
            headers.append((name, value))

        if connection_tokens is None or not any(
            token.lower() == "upgrade" for token in connection_tokens
        ):
            raise RemoteProtocolError(
                "Missing header, 'Connection: Upgrade'", event_hint=RejectConnection()
            )
        if upgrade.lower() != WEBSOCKET_UPGRADE:
            raise RemoteProtocolError(
                f"Missing header, 'Upgrade: {WEBSOCKET_UPGRADE.decode()}'",
                event_hint=RejectConnection(),
            )
        accept_token = base64.b64encode(
            hashlib.sha1(self._nonce + ACCEPT_GUID).digest())
        if accept != accept_token:
            raise RemoteProtocolError("Bad accept token", event_hint=RejectConnection())
        if subprotocol is not None:
            if subprotocol not in self._initiating_request.subprotocols:
                raise RemoteProtocolError(
                    f"unrecognized subprotocol {subprotocol}",
                    event_hint=RejectConnection(),
                )
        extensions = client_extensions_handshake(accepts, self._initiating_request.extensions)

        self._connection = Connection(
            ConnectionType.CLIENT if self.client else ConnectionType.SERVER,
            extensions,
            self._h11_connection.trailing_data[0],
        )
        self._state = ConnectionState.OPEN
        return AcceptConnection(
            extensions=extensions, extra_headers=headers, subprotocol=subprotocol
        )

    def __repr__(self) -> str:
        return "{}(client={}, state={})".format(
            self.__class__.__name__, self.client, self.state
        )


class WSConnection:
    def __init__(self, connection_type: ConnectionType) -> None:
        self.client = connection_type is ConnectionType.CLIENT
        self.handshake = H11Handshake(connection_type)
        self.connection: Optional[Connection] = None

    @property
    def state(self) -> ConnectionState:
        if self.connection is None:
            return self.handshake.state
        return self.connection.state

    def send(self, event: Event) -> bytes:
        data = b""
        if self.connection is None:
            data += self.handshake.send(event)
            self.connection = self.handshake.connection
        else:
            data += self.connection.send(event)
        return data

    def receive_data(self, data: Optional[bytes]) -> None:
        if self.connection is None:
            self.handshake.receive_data(data)
            self.connection = self.handshake.connection
        else:
            self.connection.receive_data(data)

    def events(self) -> Generator[Event, None, None]:
        yield from self.handshake.events()
        if self.connection is not None:
            yield from self.connection.events()


# MIT License
# 
# Copyright (c) 2021 Miguel Grinberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.




class SimpleWebsocketError(RuntimeError):
    pass


class ConnectionError(SimpleWebsocketError):
    def __init__(self, status_code=None):  # pragma: no cover
        self.status_code = status_code
        super().__init__(f'Connection error: {status_code}')


class ConnectionClosed(SimpleWebsocketError):
    def __init__(self, reason=CloseReason.NO_STATUS_RCVD, message=None):
        self.reason = reason
        self.message = message
        super().__init__(f'Connection closed: {reason} {message or ""}')


class Base:
    def __init__(self, sock=None, connection_type=None, receive_bytes=4096,
                 ping_interval=None, max_message_size=None,
                 thread_class=None, event_class=None, selector_class=None):
        #: The name of the subprotocol chosen for the WebSocket connection.
        self.subprotocol = None

        self.sock = sock
        self.receive_bytes = receive_bytes
        self.ping_interval = ping_interval
        self.max_message_size = max_message_size
        self.pong_received = True
        self.input_buffer = []
        self.incoming_message = None
        self.incoming_message_len = 0
        self.connected = False
        self.is_server = (connection_type == ConnectionType.SERVER)
        self.close_reason = CloseReason.NO_STATUS_RCVD
        self.close_message = None

        if thread_class is None:
            import threading
            thread_class = threading.Thread
        if event_class is None:  # pragma: no branch
            import threading
            event_class = threading.Event
        if selector_class is None:
            selector_class = selectors.DefaultSelector
        self.selector_class = selector_class
        self.event = event_class()

        self.ws = WSConnection(connection_type)
        self.handshake()

        if not self.connected:  # pragma: no cover
            raise ConnectionError()
        self.thread = thread_class(target=self._thread)
        self.thread.name = self.thread.name.replace(
            '(_thread)', '(simple_websocket.Base._thread)')
        self.thread.start()

    def handshake(self):  # pragma: no cover
        # to be implemented by subclasses
        pass

    def send(self, data):
        """Send data over the WebSocket connection.

        :param data: The data to send. If ``data`` is of type ``bytes``, then
                     a binary message is sent. Else, the message is sent in
                     text format.
        """
        if not self.connected:
            raise ConnectionClosed(self.close_reason, self.close_message)
        if isinstance(data, bytes):
            out_data = self.ws.send(Message(data=data))
        else:
            out_data = self.ws.send(TextMessage(data=str(data)))
        self.sock.send(out_data)

    def receive(self, timeout=None):
        """Receive data over the WebSocket connection.

        :param timeout: Amount of time to wait for the data, in seconds. Set
                        to ``None`` (the default) to wait indefinitely. Set
                        to 0 to read without blocking.

        The data received is returned, as ``bytes`` or ``str``, depending on
        the type of the incoming message.
        """
        while self.connected and not self.input_buffer:
            if not self.event.wait(timeout=timeout):
                return None
            self.event.clear()
        try:
            return self.input_buffer.pop(0)
        except IndexError:
            pass
        if not self.connected:  # pragma: no cover
            raise ConnectionClosed(self.close_reason, self.close_message)

    def close(self, reason=None, message=None):
        """Close the WebSocket connection.

        :param reason: A numeric status code indicating the reason of the
                       closure, as defined by the WebSocket specification. The
                       default is 1000 (normal closure).
        :param message: A text message to be sent to the other side.
        """
        if not self.connected:
            raise ConnectionClosed(self.close_reason, self.close_message)
        out_data = self.ws.send(CloseConnection(
            reason or CloseReason.NORMAL_CLOSURE, message))
        try:
            self.sock.send(out_data)
        except BrokenPipeError:  # pragma: no cover
            pass
        self.connected = False

    def choose_subprotocol(self, request):  # pragma: no cover
        del request
        # The method should return the subprotocol to use, or ``None`` if no
        # subprotocol is chosen. Can be overridden by subclasses that implement
        # the server-side of the WebSocket protocol.
        return None

    def _thread(self):
        sel = None
        if self.ping_interval:
            next_ping = time() + self.ping_interval
            sel = self.selector_class()
            try:
                sel.register(self.sock, selectors.EVENT_READ, True)
            except ValueError:  # pragma: no cover
                self.connected = False

        while self.connected:
            try:
                if sel:
                    now = time()
                    if next_ping <= now or not sel.select(next_ping - now):
                        # we reached the timeout, we have to send a ping
                        if not self.pong_received:
                            self.close(reason=CloseReason.POLICY_VIOLATION,
                                       message='Ping/Pong timeout')
                            self.event.set()
                            break
                        self.pong_received = False
                        self.sock.send(self.ws.send(Ping()))
                        next_ping = max(now, next_ping) + self.ping_interval
                        continue
                in_data = self.sock.recv(self.receive_bytes)
                if len(in_data) == 0:
                    raise OSError()
                self.ws.receive_data(in_data)
                self.connected = self._handle_events()
            except (OSError, ConnectionResetError,
                    LocalProtocolError):  # pragma: no cover
                self.connected = False
                self.event.set()
                break
        sel.close() if sel else None
        self.sock.close()

    def _handle_events(self):
        keep_going = True
        out_data = b''
        for event in self.ws.events():
            try:
                if isinstance(event, Request):
                    self.subprotocol = self.choose_subprotocol(event)
                    out_data += self.ws.send(AcceptConnection(
                        subprotocol=self.subprotocol,
                        extensions=[PerMessageDeflate()]))
                elif isinstance(event, CloseConnection):
                    if self.is_server:
                        out_data += self.ws.send(event.response())
                    self.close_reason = event.code
                    self.close_message = event.reason
                    self.connected = False
                    self.event.set()
                    keep_going = False
                elif isinstance(event, Ping):
                    out_data += self.ws.send(event.response())
                elif isinstance(event, Pong):
                    self.pong_received = True
                elif isinstance(event, (TextMessage, BytesMessage)):
                    self.incoming_message_len += len(event.data)
                    if self.max_message_size and \
                            self.incoming_message_len > self.max_message_size:
                        out_data += self.ws.send(CloseConnection(
                            CloseReason.MESSAGE_TOO_BIG, 'Message is too big'))
                        self.event.set()
                        keep_going = False
                        break
                    if self.incoming_message is None:
                        # store message as is first
                        # if it is the first of a group, the message will be
                        # converted to bytearray on arrival of the second
                        # part, since bytearrays are mutable and can be
                        # concatenated more efficiently
                        self.incoming_message = event.data
                    elif isinstance(event, TextMessage):
                        if not isinstance(self.incoming_message, bytearray):
                            # convert to bytearray and append
                            self.incoming_message = bytearray(
                                (self.incoming_message + event.data).encode())
                        else:
                            # append to bytearray
                            self.incoming_message += event.data.encode()
                    else:
                        if not isinstance(self.incoming_message, bytearray):
                            # convert to mutable bytearray and append
                            self.incoming_message = bytearray(
                                self.incoming_message + event.data)
                        else:
                            # append to bytearray
                            self.incoming_message += event.data
                    if not event.message_finished:
                        continue
                    if isinstance(self.incoming_message, (str, bytes)):
                        # single part message
                        self.input_buffer.append(self.incoming_message)
                    elif isinstance(event, TextMessage):
                        # convert multi-part message back to text
                        self.input_buffer.append(
                            self.incoming_message.decode())
                    else:
                        # convert multi-part message back to bytes
                        self.input_buffer.append(bytes(self.incoming_message))
                    self.incoming_message = None
                    self.incoming_message_len = 0
                    self.event.set()
                else:  # pragma: no cover
                    pass
            except LocalProtocolError:  # pragma: no cover
                out_data = b''
                self.event.set()
                keep_going = False
        if out_data:
            self.sock.send(out_data)
        return keep_going


class Server(Base):
    """This class implements a WebSocket server.

    Instead of creating an instance of this class directly, use the
    ``accept()`` class method to create individual instances of the server,
    each bound to a client request.
    """
    def __init__(self, environ, subprotocols=None, receive_bytes=4096,
                 ping_interval=None, max_message_size=None, thread_class=None,
                 event_class=None, selector_class=None):
        self.environ = environ
        self.subprotocols = subprotocols or []
        if isinstance(self.subprotocols, str):
            self.subprotocols = [self.subprotocols]
        self.mode = 'unknown'
        sock = None
        if 'werkzeug.socket' in environ:
            # extract socket from Werkzeug's WSGI environment
            sock = environ.get('werkzeug.socket')
            self.mode = 'werkzeug'
        elif 'gunicorn.socket' in environ:
            # extract socket from Gunicorn WSGI environment
            sock = environ.get('gunicorn.socket')
            self.mode = 'gunicorn'
        elif 'eventlet.input' in environ:  # pragma: no cover
            # extract socket from Eventlet's WSGI environment
            sock = environ.get('eventlet.input').get_socket()
            self.mode = 'eventlet'
        elif environ.get('SERVER_SOFTWARE', '').startswith(
                'gevent'):  # pragma: no cover
            # extract socket from Gevent's WSGI environment
            wsgi_input = environ['wsgi.input']
            if not hasattr(wsgi_input, 'raw') and hasattr(wsgi_input, 'rfile'):
                wsgi_input = wsgi_input.rfile
            if hasattr(wsgi_input, 'raw'):
                sock = wsgi_input.raw._sock
                try:
                    sock = sock.dup()
                except NotImplementedError:
                    pass
                self.mode = 'gevent'
        if sock is None:
            raise RuntimeError('Cannot obtain socket from WSGI environment.')
        super().__init__(sock, connection_type=ConnectionType.SERVER,
                         receive_bytes=receive_bytes,
                         ping_interval=ping_interval,
                         max_message_size=max_message_size,
                         thread_class=thread_class, event_class=event_class,
                         selector_class=selector_class)

    @classmethod
    def accept(cls, environ, subprotocols=None, receive_bytes=4096,
               ping_interval=None, max_message_size=None, thread_class=None,
               event_class=None, selector_class=None):
        """Accept a WebSocket connection from a client.

        :param environ: A WSGI ``environ`` dictionary with the request details.
                        Among other things, this class expects to find the
                        low-level network socket for the connection somewhere
                        in this dictionary. Since the WSGI specification does
                        not cover where or how to store this socket, each web
                        server does this in its own different way. Werkzeug,
                        Gunicorn, Eventlet and Gevent are the only web servers
                        that are currently supported.
        :param subprotocols: A list of supported subprotocols, or ``None`` (the
                             default) to disable subprotocol negotiation.
        :param receive_bytes: The size of the receive buffer, in bytes. The
                              default is 4096.
        :param ping_interval: Send ping packets to clients at the requested
                              interval in seconds. Set to ``None`` (the
                              default) to disable ping/pong logic. Enable to
                              prevent disconnections when the line is idle for
                              a certain amount of time, or to detect
                              unresponsive clients and disconnect them. A
                              recommended interval is 25 seconds.
        :param max_message_size: The maximum size allowed for a message, in
                                 bytes, or ``None`` for no limit. The default
                                 is ``None``.
        :param thread_class: The ``Thread`` class to use when creating
                             background threads. The default is the
                             ``threading.Thread`` class from the Python
                             standard library.
        :param event_class: The ``Event`` class to use when creating event
                            objects. The default is the `threading.Event``
                            class from the Python standard library.
        :param selector_class: The ``Selector`` class to use when creating
                               selectors. The default is the
                               ``selectors.DefaultSelector`` class from the
                               Python standard library.
        """
        return cls(environ, subprotocols=subprotocols,
                   receive_bytes=receive_bytes, ping_interval=ping_interval,
                   max_message_size=max_message_size,
                   thread_class=thread_class, event_class=event_class,
                   selector_class=selector_class)

    def handshake(self):
        in_data = b'GET / HTTP/1.1\r\n'
        for key, value in self.environ.items():
            if key.startswith('HTTP_'):
                header = '-'.join([p.capitalize() for p in key[5:].split('_')])
                in_data += f'{header}: {value}\r\n'.encode()
        in_data += b'\r\n'
        self.ws.receive_data(in_data)
        self.connected = self._handle_events()

    def choose_subprotocol(self, request):
        """Choose a subprotocol to use for the WebSocket connection.

        The default implementation selects the first protocol requested by the
        client that is accepted by the server. Subclasses can override this
        method to implement a different subprotocol negotiation algorithm.

        :param request: A ``Request`` object.

        The method should return the subprotocol to use, or ``None`` if no
        subprotocol is chosen.
        """
        for subprotocol in request.subprotocols:
            if subprotocol in self.subprotocols:
                return subprotocol
        return None
